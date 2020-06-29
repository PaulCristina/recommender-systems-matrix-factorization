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
detect_number_omp_threads()
options("rsparse::rsparse_omp_threads" = 2)

help(package = "rsparse")

# Time counter
tic.clearlog()
tic("00. Total running time")

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
# glipmse data
ratingmatSparse[1:10, 1:10]

#------------------------------------------------------------------------------------------------------------#
### 5.0 RUN MODEL ----

# get bet parameters from cross-validation
cv_results <- fread(paste0(getwd(),"/R/data/processed/tunning_results.csv"))
alpha <- cv_results$alpha[cv_results$value == min(cv_results$value)]
lambda <- cv_results$lambda[cv_results$value == min(cv_results$value)]
rank <- cv_results$rank[cv_results$value == min(cv_results$value)]

# confidence function
lin_conf <- function(x, alpha) {
  x_confidence <- x
  stopifnot(inherits(x, "sparseMatrix"))
  x_confidence@x = 1 + alpha * x@x
  return(x_confidence)
}

RhpcBLASctl::blas_set_num_threads(1)
# convergence parameters
n_iter_max = 50L
convergence_tol = 0.01

# Time counter
tic("05. Running time for model")

# Initialize model
model <- rsparse::WRMF$new(rank = rank, lambda = lambda, feedback = 'implicit')
set.seed(1)
# Conf. matrices
ratingmatSparse_conf <- lin_conf(ratingmatSparse, alpha)
# Fit
fit <- model$fit_transform(ratingmatSparse_conf,
                           n_iter = n_iter_max,
                           convergence_tol = convergence_tol)
# Item embeddings
item_embeddings <-  model$components
print(paste0("Dimmensions for embeddings: rows ", dim(item_embeddings)[1],
             ", columns ", dim(item_embeddings)[2]))
# Make a prediction
print(paste0("Recommendation number: ", number_of_Reco))
new_user_predictions <- model$predict(ratingmatSparse_conf,
                                      k = number_of_Reco,
                                      not_recommend = ratingmatSparse_conf)
str(new_user_predictions)

# end timer
toc(log = TRUE)


#------------------------------------------------------------------------------------------------------------#
### 6.0 MAKE RECOMMENDATIONS WITH ALS ----

# Time counter
tic("06. Reco with Als time")

print(paste0("Get predictions:...", now()))
# convert recommender object to readable list
df.recom_list <- data.table(new_user_predictions, keep.rownames=TRUE)

# convert viewer id to integer
df.recom_list[, rn := as.integer(substring(rn, 2))]
setnames(df.recom_list, "rn", "userIdDense")
df.recom_list$userId <- df.userDense$userId[match(as.integer(df.recom_list$userIdDense),
                                                    df.userDense$userIdDense)]

# retrive columns that are recommendations
recom_list_names <- names(df.recom_list[,2:(number_of_Reco+1)])
recom_list_names

# retrive movies names
for(i in recom_list_names){
  df.recom_list[[paste(i, 'reco', sep="_")]] <- df.movies$title[match(as.integer(df.recom_list[[i]]),
                                                                      df.movies$movieIdDense)]
}
df.recom_list <- setDT(df.recom_list)
# delete obsolute columns
cols.to.del <- NULL
cols.to.del <- names(df.recom_list[,1:(number_of_Reco+1)])
df.recom_list[, (cols.to.del) := NULL]


# transform recommendations to list
cols.to.del <- NULL
cols.to.del <- names(df.recom_list[,2:(number_of_Reco+1)])
df.recom_list[, reco := do.call(paste, c(.SD, sep = ";")), .SDcols = cols.to.del]
df.recom_list[, (cols.to.del) := NULL]


# save as parquet
path <- paste0(getwd(),"/R/data/processed/reco_als.parquet")
write_parquet(df.recom_list, path)

# end timer
toc(log = TRUE)

#------------------------------------------------------------------------------------------------------------#
### 7.0 MAKE RECOMMENDATIONS  WITH SIMILAR ITEMS----

# Time counter
tic("07. Reco with similar items time")

# transpose for convenience of cosine distance calculation
movie_emb <- t(model$components)
# make empty list to hold recommendations
i <- NULL
datalist <- vector("list", nrow(df.movies))  # initialize the results list.
# empty dataframe for recommendations
recoCols <- paste0("V", 1:(number_of_Reco+1))
df.recom_sims <- as.data.frame(matrix(,0, length(recoCols)))
names(df.recom_sims) <- recoCols


for(i in df.movies$movieIdDense){
  chosenCam <- i
  cam_query <- movie_emb[paste0("m", chosenCam), , drop = FALSE]
  cam_query_sim <- text2vec::sim2(cam_query, movie_emb, method = "cosine")
  # flatten 1-row matrix
  cam_query_sim = cam_query_sim[1, ]
  # check similar artists
  names <- attributes(sort(cam_query_sim, decreasing = T)[1:(number_of_Reco+1)])
  movie_sim <- data.table(value = sort(cam_query_sim, decreasing = T)[1:(number_of_Reco+1)],
                          movieIdDense = names$names)
  movie_sim[, movieIdDense := as.integer(substring(movieIdDense, 2))]
  datalist[[i]] <- t(movie_sim$movieIdDense)
  df.recom_sims <- rbind(df.recom_sims, datalist[[i]])
}


# retrive movies true names
for(i in names(df.recom_sims)){
  df.recom_sims[[paste(i, 'reco', sep="_")]] <- df.movies$title[match(as.integer(df.recom_sims[[i]]),
                                                                      df.movies$movieIdDense)]
}
df.recom_sims <- setDT(df.recom_sims)
# delete obsolute columns
cols.to.del <- NULL
cols.to.del <- names(df.recom_sims[,1:(number_of_Reco+1)])
df.recom_sims[, (cols.to.del) := NULL]

# transform recommendations to list
cols.to.del <- NULL
cols.to.del <- names(df.recom_sims[,2:(number_of_Reco+1)])
df.recom_sims[, reco := do.call(paste, c(.SD, sep = ";")), .SDcols = cols.to.del]
df.recom_sims[, (cols.to.del) := NULL]
setnames(df.recom_sims, names(df.recom_sims), c("movies", "similar"))

# save as parquet
path <- paste0(getwd(),"/R/data/processed/reco_items.parquet")
write_parquet(df.recom_sims, path)

# end timer
toc(log = TRUE)
