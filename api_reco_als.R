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
options("rsparse::rsparse_omp_threads" = 1)
help(package = "RestRserve")


#------------------------------------------------------------------------------------------------------------#
### 3.0 CREATE APPLICATION----
app = Application$new(
  content_type = "application/json"
)

#------------------------------------------------------------------------------------------------------------#
### 4.0 CREATE FUNCTION WHICH WILL HANDLE REQUESTS ----
reco_handler = function(request, response) {



  # arguments needed
  userReco <- as.integer(request$parameters_query[["x"]])
  # EX:
  # userReco <- as.integer(1)

  # retrive recommendation data
  url <- "https://raw.githubusercontent.com/PaulCristina/recommender-systems-matrix-factorization/master/R/data/processed/reco_als.csv"
  df.reco <- fread(url)


  if (!is.numeric(userReco) || !userReco %in% df.reco$userId) {
    raise(HTTPError$bad_request(body = "Incorect user"))
  }

  #' recommendations as json
  df.recom_user <- df.reco[userId == userReco,]

  if (nrow(df.recom_user) != 1) {
    raise(HTTPError$bad_request(body = "No data"))
  }

  # tranform to json
  recom_json <- strsplit(df.recom_user$reco, ";")[[1]]
  recom_json <- jsonlite::toJSON(recom_json, auto_unbox = TRUE, pretty = TRUE)


  if (jsonlite::validate(recom_json) != TRUE) {
    raise(HTTPError$bad_request(body = "Incorect movie list"))
  }

  ### set body
  response$body = recom_json
  response$status_code = 200L

}

#------------------------------------------------------------------------------------------------------------#
### 5.0 REGISTER ENDPOINT ----
app$add_get(path = "/reco",
            FUN = reco_handler)


#------------------------------------------------------------------------------------------------------------#
### 6.0 ADD OPENAPI DESCIPTION AND SWAGGER UI ----

app$add_openapi(
  path = paste0(getwd(),"/api_reco_als.yaml"),
  file_path = paste0(getwd(),"/api_reco_als.yaml")
)


# see details on https://swagger.io/tools/swagger-ui/
app$add_swagger_ui(path = "/doc",
                   path_openapi = paste0(getwd(),"/api_reco_als.yaml"),
                   use_cdn = TRUE)

#------------------------------------------------------------------------------------------------------------#
### 7.0 START THE APP ----
backend = BackendRserve$new()
backend$start(app, http_port = 8585)

### Check it works ----
# curl localhost:8585/reco?x=1
# Check out a swagger UI in the browser: localhost:8585/doc
