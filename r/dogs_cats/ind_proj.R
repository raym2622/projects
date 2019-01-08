############################################################
# Individual Project                                       #
#                                                          #
#                                                          #
# This script imports .csv data from:                      #                         
# https://catalog.data.gov/dataset/seattle-pet-licenses    #
# and                                                      #
# https://catalog.data.gov/dataset/seattle-zip-codes-ebab5 #
#                                                          #
# Author: Rui Ma (mrui@umich.edu)                          #
# Date:   12/15/17                                         #
############################################################

library(dplyr)
library(stringr)
library(tibble)
library(tidyr)
library(ggmap)
library(knitr)
library(zipcode)
data(zipcode)
# import data sets as tibbles
licenses = as_tibble(read.csv("Seattle_Pet_Licenses.csv"))
zip_codes = as_tibble(read.csv("Zip_Codes.csv"))

# select and rename variables
licenses = licenses %>% 
    select(year = License.Issue.Date, id = License.Number, species = Species,
           zip = ZIP.Code)

# select and rename variables
zip_codes = zip_codes %>%
    select(zip = ZIP, area = Shape_Area) %>%
    group_by(zip) %>% 
    distinct(zip, .keep_all = TRUE) %>%
    filter(area != 0)

# data cleaning for zip codes
clean_zip = function(x){
    x = gsub(" ", "", x)    
}

# this is a wrapper to an apply call
clean_all_zip = function(x){
    sapply(x, clean_zip)
}

# data cleaning for dates
parse_date = function(x){
    x = gsub(" ", "", x)
    str_sub(x,-4,-1)
}

# this is a wrapper to an apply call
get_year = function(x){
    sapply(x, parse_date)
}

# get rid of NA zip_codes and get 2013-2016 licenses
licenses = licenses %>% 
    mutate(zip = clean_all_zip(zip)) %>%
    filter(zip != "") %>%
    mutate(year = get_year(year)) %>%
    filter(year >= 2013)

# merge licenses and zip_codes by the common column "zip"
# automatically get rid of all owners whose zip codes don't belong in the 
# Greater Seattle area
licenses_zip = merge(licenses, zip_codes, by = "zip")
licenses_zip = licenses_zip %>%
    select(year, id, species, zip, area) %>%
    arrange(year)

# check if there's any NA's
sum(is.na(licenses_zip$area))

# distribution of dogs
dog_count = licenses_zip %>%
    filter(species == "Dog") %>%
    group_by(zip, area) %>%
    summarize(count = n()) %>%
    ungroup() %>%
    transmute(zip, ratio = count / (area/1000000), count, area) %>%
    merge(zipcode, by = 'zip') %>%
    select(-c(city, state)) %>%
    arrange(desc(ratio))

# distribution of cats
cat_count = licenses_zip %>%
    filter(species == "Cat") %>% 
    group_by(zip, area) %>%
    summarize(count = n()) %>%
    ungroup() %>%
    transmute(zip = zip, ratio = count / (area/1000000), count, area) %>%
    merge(zipcode, by = 'zip') %>%
    select(-c(city, state)) %>%
    arrange(desc(ratio))

# generate nice tables
t1 = dog_count[1:3, 1:4]
t2 = cat_count[1:3, 1:4]
kable(list(t1,t2), caption = "Distribution of Dogs vs. Distribution of Cats",
      format = "latex", booktabs = T) %>%
    kable_styling(latex_options = "hold_position")

# use ggmap to plot the maps
map = get_map(location = 'Seattle', zoom = 11)
ggmap(map) + geom_point(
    aes(x=longitude, y=latitude, fill = ratio), 
    data=dog_count[c(1:3),], 
    shape = 23,
    alpha=.7,
    size = c(9, 7, 5),
    na.rm = T)  + 
    scale_fill_continuous(low = "dark blue", high = "orange")

ggmap(map) + geom_point(
    aes(x=longitude, y=latitude, fill = ratio), 
    data=cat_count[c(1:3),], 
    shape = 22,
    alpha=.7,
    size = c(8, 6, 5),
    na.rm = T)  + 
    scale_fill_continuous(low = "dark blue", high = "red")

