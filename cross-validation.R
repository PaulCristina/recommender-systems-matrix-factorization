#------------------------------------------------------------------------------------------------------------#
### 1.0 REMOVE OBJECTS ----
# .rs.restartR()
rm(list = ls(all.names=TRUE))
gc()
options(scipen = 999)

start.overall.time <- Sys.time()
print("###########################################################")
print("###########################################################")
print("Script version is: 0.0.1")
print(paste0("Start process: ", start.overall.time))

# Set project path
setwd("/recommender-systems-matrix-factorization")
#------------------------------------------------------------------------------------------------------------#
### 2.0 LOAD LIBRARY'S ----
# Load libraries
source("R/libraries/librariesUsed.R")
sessionInfo()
options("rsparse::rsparse_omp_threads" = 2)

help(package = "rsparse")

#------------------------------------------------------------------------------------------------------------#
### 3.0 LOAD DATA ----

# download data
# https://grouplens.org/datasets/movielens/
data_dir <- paste0(getwd(),"/R/data/raw")
df.movies <- fread(file.path(data_dir, "movies.csv.gz"))
df.ratings <- fread(file.path(data_dir, "ratings.csv.gz"))
glimpse(df.ratings)
glimpse(df.movies)


# number of views a movie must have in order
# to be included in the recommendations
movieViews <- 100
# number of views a user must have in order
# to be included in the recommendations
movieWatched <- 10
# recommendations needed
number_of_Reco <- 10


#------------------------------------------------------------------------------------------------------------#
### 4.0 PROCESS DATA ----

# number of visualization for each movie
df.ratings[, number_of_visualization_per_movie := .N, by = movieId]
# remove movies with less then camsViews views
df.ratings <- df.ratings[number_of_visualization_per_movie >= movieViews, ]

# number of movies a user has watched
df.ratings <- df.ratings[,views := .N, by = userId]
# remove movies with less then camsViews views
df.ratings <- df.ratings[views >= movieWatched, ]

# convert rating to implicit
df.ratings[, rating := ifelse(rating >= 4, 1, NA)]
df.ratings <- df.ratings[rating == 1,]

# generate new movie id
df.ratings[, movieIdDense := .GRP, by = movieId]

# generate new user id
df.ratings[, userIdDense := .GRP, by = userId]

# add new id to movie data
df.moviesDense <- df.ratings[,.(movieIdDense = unique(movieIdDense)),
                            by = movieId]
df.movies <- merge(df.movies, df.moviesDense,
                   by = "movieId", all.x = TRUE, all.y = FALSE)
# remove movies without id
df.movies <- na.omit(df.movies)

# dataset for unique users
df.userDense <- df.ratings[,.(userIdDense = unique(userIdDense)),
                             by = userId]

# create ratings matrix. Rows = userIdDense, Columns = movieIdDense
ratingmatSparse <- sparseMatrix(i = df.ratings$userIdDense,
                                j = df.ratings$movieIdDense,
                                x = df.ratings$rating,
                                dims = c(length(unique(df.ratings$userIdDense)),
                                         length(unique(df.ratings$movieIdDense))),
                                dimnames = list(paste("u", 1:length(unique(df.ratings$userIdDense)), sep = ""),
                                                paste("m", 1:length(unique(df.ratings$movieIdDense)), sep = "")))
ratingmatSparse[1:10, 1:10]

#------------------------------------------------------------------------------------------------------------#
### 5.0 TRAIN/TEST SPLIT ----

# select subset of users
N_CV <- 30000L
cv_uid <- sample(nrow(df.userDense), N_CV)

# split into train/test
X_train <- ratingmatSparse[-cv_uid, ]
X_test  <- ratingmatSparse[cv_uid, ]
dim(X_train)
dim(X_test)

# split rating for testing data into history and actual validation
# for each user take half of the ratings and treat them as history
# the rest as future ratings
temp <- as(X_test, "TsparseMatrix")
temp <- data.table(i = temp@i, j = temp@j, x = temp@x)

temp <- temp %>%
  group_by(i) %>%
  mutate(ct = length(j),
         history =
           sample(c(TRUE, FALSE), ct, replace = TRUE, prob = c(.5, .5))) %>%
  select(-ct)

X_test_history <- temp %>% filter(history == TRUE)
X_test_future <- temp %>% filter(history == FALSE)


# convert to sparse matrices
X_test_history <- sparseMatrix(i = X_test_history$i,
                               j = X_test_history$j,
                               x = X_test_history$x,
                               dims = dim(X_test),
                               dimnames = dimnames(X_test),
                               index1 = FALSE)

X_test_future <- sparseMatrix(i = X_test_future$i,
                              j = X_test_future$j,
                              x = X_test_future$x,
                              dims = dim(X_test),
                              dimnames = dimnames(X_test),
                              index1 = FALSE)



#------------------------------------------------------------------------------------------------------------#
### 6.0 TUNING PARAMETRES  ----

# define linear confidence functions
lin_conf <- function(x, alpha) {
  x_confidence <- x
  stopifnot(inherits(x, "sparseMatrix"))
  x_confidence@x = 1 + alpha * x@x
  return(x_confidence)
}

futile.logger::flog.threshold(futile.logger::ERROR)
RhpcBLASctl::blas_set_num_threads(1)

# convergence parameters
n_iter_max = 50L
convergence_tol = 0.01

trace = NULL
n_threads = 8L

# Empty vector to throw results into
scores <-  vector("list", nrow(grid))

# hyperparameters grid
# (a very long tunning run)takes around 1h for 120 runs)
grid <- expand.grid(alpha = c(.01, .05, .1, 1),
                    rank = c(8, 16, 32, 40, 80),
                    lambda = c(.01, .05, .1, 1, 10, 15))

for(k in seq_len(nrow(grid))) {
  alpha = grid$alpha[[k]]
  rank = grid$rank[[k]]
  lambda = grid$lambda[[k]]

  # initialize the model
  model <- rsparse::WRMF$new(rank = rank,
                             lambda = lambda,
                             feedback = 'implicit')

  # conf. matrices
  X_train_conf<- lin_conf(X_train, alpha)
  X_test_history_conf <- lin_conf(X_test_history, alpha)

  # fit model
  user_embeddings = model$fit_transform(X_train_conf,
                                        n_iter = n_iter_max,
                                        convergence_tol = convergence_tol,
                                        n_threads = n_threads)
  # store trace
  grid_trace = attr(user_embeddings, "trace")
  grid_trace$param_set = sprintf("alpha=%.3f; rank=%d", alpha, rank, lambda)
  trace = c(trace, list(grid_trace))

  # extract score
  score =  attr(user_embeddings, "trace")

  score$alpha = alpha
  score$lambda = lambda
  score$rank = rank

  # add to list
  scores[[k]] <-  score

  # clean up
  rm(alpha, rank, lambda, model, score)

}
trace = rbindlist(trace)


g = ggplot(trace) +
  geom_line(aes(x = iter, y = value, col = param_set)) +
  facet_wrap( ~ scorer, scales = "free") +
  theme(legend.position="bottom")
plotly::ggplotly(g, width = 9, height = NULL)

cv_results <-  bind_rows(scores) %>%
  group_by(alpha, lambda, rank, scorer) %>%
  arrange(iter) %>%
  filter(row_number() == n()) %>%
  select(-iter) %>%
  ungroup()

fwrite(cv_results, paste0(getwd(),"/R/data/processed/tunning_results.csv"))

#------------------------------------------------------------------------------------------------------------#
### 7.0 CALCULATE ACCURACY  ----


alpha <- cv_results$alpha[cv_results$value == min(cv_results$value)]
lambda <- cv_results$lambda[cv_results$value == min(cv_results$value)]
rank <- cv_results$rank[cv_results$value == min(cv_results$value)]

# Convergence parameters
n_iter_max = 50L
convergence_tol = .01

# initialize the model
model <- rsparse::WRMF$new(rank = rank,
                           lambda = lambda,
                           feedback = 'implicit')

# conf. matrices
X_train_conf<- lin_conf(X_train, alpha)
X_test_history_conf <- lin_conf(X_test_history, alpha)

# fit model
user_embeddings = model$fit_transform(X_train_conf,
                                      n_iter = n_iter_max,
                                      convergence_tol = convergence_tol,
                                      n_threads = n_threads)

# item embeddings
item_embeddings <-  model$components

# make a prediction
new_user_predictions <- model$predict(X_test_history,
                                      k = number_of_Reco,
                                      not_recommend = X_test_history)

# Normalized Discounted Cumulative Gain(ndcg)
# https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

# empty vector to throw results into
ndcg_score <-  vector("list", nrow(X_test_history))

for(i in seq_len(nrow(X_test_history))) {
  score <- NULL
  score <- rsparse::ndcg_k(predictions = as.matrix(new_user_predictions[i, , drop = FALSE]),
                  actual = X_test_future[i, , drop = FALSE])
  # add to list
  ndcg_score[[i]] <-  score

}

(ndcg_results <-  round(mean(as.numeric(ndcg_score)), 2))
