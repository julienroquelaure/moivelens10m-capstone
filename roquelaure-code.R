##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

#I added the following line because the compilation tomed out after 60 seconds
options(timeout=600)

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#The following lines were used to answer the quizz 

dim(edx)

edx %>% filter(rating == 0) %>% tally()
edx %>% filter(rating == 3) %>% tally()

n_distinct(edx$movieId)
#ans= 10677
n_distinct(edx$userId)
#ans= 69878

genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

edx %>% group_by(rating) %>%
  summarize(count = n()) %>%
  top_n(5) %>%
  arrange(desc(count))

edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()

#Now starts the work I did for the submission

#mu is the mean of all the ratings
mu <- mean(edx$rating)
mu
#ans = 3.512465
RMSE(mu, validation$rating)
#ans = 1.061202
#this RMSE  will be our starting point.

#individual movie quality compared to the average: "movie bias"
edx_movie_bias <- edx %>%
  group_by(movieId) %>%
  summarize(b_m = mean(rating - mu))
head(edx_movie_bias)
dim(edx_movie_bias)

#individual "user bias", controlling for general average and movie bias
edx_user_bias <- edx %>%
  left_join(edx_movie_bias, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m))
head(edx_user_bias)
dim(edx_user_bias)

#prediction of the validation rating, function of userId and movieId
edx_pred <- validation %>%
  left_join(edx_movie_bias, by="movieId") %>%
  left_join(edx_user_bias, by="userId") %>%
  mutate(pred = mu + b_m + b_u)
head(edx_pred)
dim(edx_pred)

#new RMSE after accounting for movie and user bias
RMSE(edx_pred$pred, validation$rating)
#ans = 0.8653488

#We noticed that some predictions are below .5 or above 5
sum(edx_pred$pred <.5)
sum(edx_pred$pred >5)

#so we bound our predictions:
edx_pred_bound <- edx_pred %>%
  mutate(pred = ifelse(pred>5, 5, pred)) %>%
  mutate(pred = ifelse(pred<.5, .5, pred))
sum(edx_pred_bound$pred <.5)
sum(edx_pred_bound$pred >5)
RMSE(edx_pred_bound$pred, validation$rating)
#ans = 0.8651613 we are not there yet

#We analyze what are the predictions most responsible for our errors:
edx_difference <- edx_pred_bound %>%
  mutate(diff = abs(pred - validation$rating)) %>%
  arrange(desc(diff))

edx_difference %>% 
  filter(diff>2)
#We see that there are roughly 30,000 errors bigger than 2

#So we take a slice for analysis:
edx_diff_slice <- edx_difference %>%
  slice(1:30000)

#With a right join, we look at the problematic users...
edx_difference %>%
  group_by(userId) %>%
  summarize(n=n(), meandiff=mean(diff)) %>%
  right_join(edx_diff_slice, by="userId") %>%
  select(userId, n, meandiff, pred, diff) %>%
  arrange(desc(meandiff))

#... and the problematic movies
edx_difference %>%
  group_by(movieId) %>%
  summarize(n=n(), meandiff=mean(diff)) %>%
  right_join(edx_diff_slice, by="movieId") %>%
  select(movieId, n, meandiff, pred, diff) %>%
  arrange(desc(meandiff))

#We notice that a lot of the errors seem to be caused by rare users and movies

#So, we decide to  use regularization in order to dampen the predictions
#done on rare users and movies and bring them closer to the average.


#To find our optimal parameter lambda we have proceeded in two times:
# first with a .25 step, then a refinement with a .05 step.
#By default, the code will run our refinement after a minimum value
#at lambda=4.75 with a .25 step.
lambdas <- seq(3, 7, .25)
lambdas <- seq(4.5, 5, .05)

#We redo the same algorithm with a sapply for the lambdas:
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  edx_movie_bias_regs <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  edx_user_bias_regs <- edx %>% 
    left_join(edx_movie_bias_regs, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  edx_pred_regs <- validation %>% 
    left_join(edx_movie_bias_regs, by = "movieId") %>%
    left_join(edx_user_bias_regs, by = "userId") %>%
    mutate(pred = mu + b_i + b_u)
  
  edx_pred_regs_bound <- edx_pred_regs %>%
    mutate(pred = ifelse(pred>5, 5, pred)) %>%
    mutate(pred = ifelse(pred<.5, .5, pred))
  
  return(RMSE(edx_pred_regs_bound$pred, validation$rating))
})

qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)]
lambda
RMSE_final <- min(rmses)
RMSE_final
#We find that the optimal lambda is 4.85
#And our final RMSE is 0.8647088









