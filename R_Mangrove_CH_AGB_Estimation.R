#### ESTIMATING MANGROVE CANOPY HEIGHT AND ABOVEGROUND BIOMASS USING MULTISOURCE REMOTE SENSING ####

# Data compiled and code written by Novia Arinda Pradisty
# Published date: 15-10-2025

# Based on the publication of:
# Pradisty NA, Schlund M, Horstman EM, Willemen L. Under review. Estimating canopy height and aboveground biomass in tropical mangrove restoration areas through multisource remote sensing. Ecological Informatics.

# The model development data (i.e. RF model data BP.tif) consists of 15-m spatial resolution canopy height (Height.q95) and aboveground biomass (AGB.Chave) layers as reference data,
# complemented by multiple layers of multisource remote sensing data (from Sentinel-1 (S1), Sentinel-2 (S2), SAOCOM-1 (SAO) and Copernicus GLO-30 DEM).
# Non-proprietary and non-public data and is available in Zenodo: (DOI: 10.5281/zenodo.17359993). 
# SAOCOM-1 data can generally be requested and accessed at the SAOCOM Catalog of the Argentinean National Commission on Space Activities (CONAE) (https://catalog.saocom.conae.gov.ar/catalog/). 
# Sentinel-1 and Sentinel-2 data can be publicly accessed at Copernicus Data Space Ecosystem (CDSE) of the European Space Agency (ESA) (https://dataspace.copernicus.eu/).
# BP = Budeng-Perancak; TA = Tahura Ngurah Rai

# Clear all objects from the workspace
# rm(list = ls())

# Load required R packages
# Install packages by, for instance:
# install.packages(c("sf", "terra", "raster"))

# Note 1: Change 'your/path' to your own path.
# Note 2: Run ggsave, save.csv, writeVector or saveRDS line by removing/excluding the hashtag (#). 

library(sf)           # Spatial data handling
library(terra)        # Raster data processing
library(raster)       # Raster data (backward compatibility)
library(tidyverse)    # Data manipulation and visualization
library(tidymodels)   # Modeling framework
library(spatialsample)# Spatial cross-validation
library(randomForest) # Random forest implementation
library(purrr)        # Functional programming
library(furrr)        # Parallel processing
library(tidysdm)      # Spatial domain modeling
library(viridis)      # Color scales
library(patchwork)    # Plot arrangement
library(vip)          # Variable importance
library(GGally)       # Correlation plots
library(ggpointdensity) # Density scatter plots
library(zen4R)        # Data repository access


# Set preferred functions to avoid conflicts
tidymodels_prefer()

##########################   DATA FOR MODEL   ###########################

# Access data from Zenodo repository
download_zenodo("10.5281/zenodo.17359993")

# Read 15-m spatial resolution canopy height - aboveground biomass (CH - AGB) data and project to UTM zone 50S
ch_agb <- terra::rast('your/path/CH_AGB_15m.tif') %>%  terra::project('EPSG:32750')
plot(ch_agb)

# List of Sentinel-1 variables: "Coherence_VH_S1","Coherence_VV_S1",
# "Entropy_VHVV_S1","Anisotropy_VHVV_S1","Alpha_VHVV_S1","VH_S1","VV_S1",
# "VHVV_Ratio_S1","VHVV_S1","sqrtVHVV_S1"

# List of Sentinel-2 variables: "Blue_S2","Green_S2","Red_S2","RedEdge1_S2","RedEdge2_S2",
# "RedEdge3_S2","NIR1_S2","RedEdge4_S2","SWIR1_S2","SWIR2_S2",
# "AFRI1600_S2","BI_S2","BITM_S2","BIXS_S2","DBSI_S2","IKAW_S2","LSWI_S2","MBI_S2",      
# "MRBVI_S2","MSI_S2","NDII_S2","NDMI_S2","NDVI_S2",'MTCI_S2','REDSI_S2', 'BI_S2', 
# "PSRI_S2","RCC_S2","RENDVI_S2","SAVI_S2","WFI_S2","MDI_S2"

# List of SAOCOM-1 variables: "HH_SAO","HV_SAO","VH_SAO","VV_SAO",
# "HHHV_SAO","sqrtHHHV_SAO","HHHV_Ratio_SAO",
# "VHVV_SAO","sqrtVHVV_SAO","VHVV_Ratio_SAO",
# "Entropy_HHHV_SAO","Anisotropy_HHHV_SAO","Alpha_HHHV_SAO",
# "Entropy_VHVV_SAO","Anisotropy_VHVV_SAO","Alpha_VHVV_SAO",
# "Coherence_HV_SAO","Coherence_HH_SAO",
# "Coherence_VH_SAO","Coherence_VV_SAO"

# Copernicus GLO-30 DEM elevation variable: "Elevation_Cop30m"

# Read multisource satellite data and project to UTM zone 50S
rr_modelbp <- terra::rast('your/path/RF model data BP.tif') %>%  terra::project('EPSG:32750')
rr_predbp <- terra::rast('your/path/Prediction data BP.tif') %>%  terra::project('EPSG:32750')
rr_predta <- terra::rast('your/path/Prediction data TA.tif') %>%  terra::project('EPSG:32750')

# Convert raster data to sf objects for spatial analysis
rr_sf <- as.data.frame(rr_modelbp, xy = TRUE) %>% 
  drop_na() %>% 
  st_as_sf(coords = c("x", "y"), crs = 32750)

rrbp_sf <- as.data.frame(rr_predbp, xy = TRUE)  %>%
  drop_na() %>%
  st_as_sf(coords = c("x", "y"), crs = 32750)

rrta_sf <- as.data.frame(rr_predta, xy = TRUE) %>%
  drop_na() %>%
  st_as_sf(coords = c("x", "y"), crs = 32750)


######################### SPATIAL CROSS-VALIDATION SETUP #########################

# Set random seed for reproducibility
set.seed(100)

# Create initial spatial data split
# First data split
rr_block_split1 <-
  spatial_initial_split(
    data=rr_sf,
    prop = 1/4, 
    buffer=45, #in meter
    strategy = spatial_block_cv)
rr_block_split1 

# Split data into training and testing sets
# First spatial partition
# Data partition for final model training and evaluation
train_data1 = training(rr_block_split1)
# Data partition for model development (performance evaluation, feature selection and hyperparameter tuning)
test_data1 = testing(rr_block_split1) 

# Create secondary spatial split for model evaluation
set.seed(100)
rr_block_split21 <-
  spatial_initial_split(
    data=st_as_sf(test_data1),
    prop = 1/4,
    buffer=45,
    strategy = spatial_block_cv)

# Further spatial split 
train_data21 = training(rr_block_split21) 
test_data21 = testing(rr_block_split21) 

# Convert to sf objects with proper coordinate reference system
train21_sf <- train_data21%>% st_as_sf(coords = c("x", "y"), crs = 32750) 
test21_sf <- test_data21  %>% st_as_sf(coords = c("x", "y"), crs = 32750)

# Create spatial cross-validation folds
spatial_folds_train21 <- spatial_block_cv(data=train_data21,v=10,repeats=10, 
                                          method='random', buffer=500) 
spatial_folds_test21 <- spatial_block_cv(data=test_data21, v=10, repeats=10,
                                         method='random',buffer=500) 
# Exemplary plot of spatial folds
autoplot(spatial_folds_train21)


# Random forest model specification for canopy height and aboveground biomass
# with default hyperparameters: rand_forest() 

rf_model_ch <- rand_forest()  %>%
  set_engine("ranger",
             keep.inbag = TRUE,  
             importance = "permutation",
             seed = 100) %>%
  set_mode("regression") 

rf_model_agb <- rand_forest()  %>%
  set_engine("ranger",
             keep.inbag = TRUE,    
             importance = "permutation",
             seed = 100) %>%
  set_mode("regression") 


################  RECIPE LIST ################
# Define preprocessing recipes for different predictor combinations

# Basic recipe using all predictors
recipe_ch <- recipe(formula=Height.q95 ~., data = rr_sf)%>% 
  step_filter(is.na(Height.q95) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","AGB.Chave")))

recipe_agb <- recipe(formula=AGB.Chave ~., data = rr_sf) %>% 
  step_filter(is.na(AGB.Chave) == FALSE) %>% 
  step_rm(any_of(c("x", "y","geometry","Height.q95")))  

# Recipe using only elevation
recipe_agb_elev <-recipe(formula=AGB.Chave ~., data = rr_sf) %>%
  step_filter(is.na(AGB.Chave) == FALSE)%>%
  step_rm(any_of(c("x", "y", "geometry","Height.q95","Blue_S2","Green_S2","Red_S2","RedEdge1_S2","RedEdge2_S2",
                   "RedEdge3_S2","NIR1_S2","RedEdge4_S2","SWIR1_S2","SWIR2_S2",
                   "AFRI1600_S2"," BI_S2","BITM_S2","BIXS_S2","DBSI_S2","IKAW_S2","LSWI_S2","MBI_S2",     
                   "MRBVI_S2","MSI_S2","NDII_S2","NDMI_S2","NDVI_S2",'MTCI_S2','REDSI_S2', 'BI_S2',    
                   "PSRI_S2","RCC_S2","RENDVI_S2","SAVI_S2","WFI_S2","MDI_S2",
                   "HH_SAO","HV_SAO","VH_SAO","VV_SAO",
                   "HHHV_SAO","sqrtHHHV_SAO","HHHV_Ratio_SAO",
                   "VHVV_SAO","sqrtVHVV_SAO","VHVV_Ratio_SAO",
                   "Entropy_HHHV_SAO","Anisotropy_HHHV_SAO","Alpha_HHHV_SAO",
                   "Entropy_VHVV_SAO","Anisotropy_VHVV_SAO","Alpha_VHVV_SAO",
                   "Coherence_HV_SAO","Coherence_HH_SAO",
                   "Coherence_VH_SAO","Coherence_VV_SAO",
                   "Coherence_VH_S1","Coherence_VV_S1",
                   "Entropy_VHVV_S1","Anisotropy_VHVV_S1","Alpha_VHVV_S1","VH_S1","VV_S1",
                   "VHVV_Ratio_S1","VHVV_S1","sqrtVHVV_S1"))) 

recipe_ch_elev <-recipe(formula=Height.q95 ~., data = rr_sf) %>%
  step_filter(is.na(Height.q95) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","AGB.Chave","Blue_S2","Green_S2","Red_S2","RedEdge1_S2","RedEdge2_S2",
                   "RedEdge3_S2","NIR1_S2","RedEdge4_S2","SWIR1_S2","SWIR2_S2",
                   "AFRI1600_S2","BI_S2","BITM_S2","BIXS_S2","DBSI_S2","IKAW_S2","LSWI_S2","MBI_S2",     
                   "MRBVI_S2","MSI_S2","NDII_S2","NDMI_S2","NDVI_S2",'MTCI_S2','REDSI_S2', 'BI_S2',    
                   "PSRI_S2","RCC_S2","RENDVI_S2","SAVI_S2","WFI_S2","MDI_S2",
                   "HH_SAO","HV_SAO","VH_SAO","VV_SAO",
                   "HHHV_SAO","sqrtHHHV_SAO","HHHV_Ratio_SAO",
                   "VHVV_SAO","sqrtVHVV_SAO","VHVV_Ratio_SAO",
                   "Entropy_HHHV_SAO","Anisotropy_HHHV_SAO","Alpha_HHHV_SAO",
                   "Entropy_VHVV_SAO","Anisotropy_VHVV_SAO","Alpha_VHVV_SAO",
                   "Coherence_HV_SAO","Coherence_HH_SAO",
                   "Coherence_VH_SAO","Coherence_VV_SAO",
                   "Coherence_VH_S1","Coherence_VV_S1",
                   "Entropy_VHVV_S1","Anisotropy_VHVV_S1","Alpha_VHVV_S1","VH_S1","VV_S1",
                   "VHVV_Ratio_S1","VHVV_S1","sqrtVHVV_S1")))

# Recipe using only Sentinel-1 and -2
recipe_agb_sentinel <-recipe(formula=AGB.Chave ~., data = rr_sf) %>%
  step_filter(is.na(AGB.Chave) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","Height.q95","HH_SAO","HV_SAO","VH_SAO","VV_SAO",
                   "HHHV_SAO","sqrtHHHV_SAO","HHHV_Ratio_SAO",
                   "VHVV_SAO","sqrtVHVV_SAO","VHVV_Ratio_SAO",
                   "Entropy_HHHV_SAO","Anisotropy_HHHV_SAO","Alpha_HHHV_SAO",
                   "Entropy_VHVV_SAO","Anisotropy_VHVV_SAO","Alpha_VHVV_SAO",
                   "Coherence_HV_SAO","Coherence_HH_SAO",
                   "Coherence_VH_SAO","Coherence_VV_SAO","Elevation_Cop30m"))) 

recipe_ch_sentinel <-recipe(formula=Height.q95 ~., data = rr_sf) %>%
  step_filter(is.na(Height.q95) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","AGB.Chave","HH_SAO","HV_SAO","VH_SAO","VV_SAO",
                   "HHHV_SAO","sqrtHHHV_SAO","HHHV_Ratio_SAO",
                   "VHVV_SAO","sqrtVHVV_SAO","VHVV_Ratio_SAO",
                   "Entropy_HHHV_SAO","Anisotropy_HHHV_SAO","Alpha_HHHV_SAO",
                   "Entropy_VHVV_SAO","Anisotropy_VHVV_SAO","Alpha_VHVV_SAO",
                   "Coherence_HV_SAO","Coherence_HH_SAO",
                   "Coherence_VH_SAO","Coherence_VV_SAO","Elevation_Cop30m"))) 

# Recipe using only Sentinel-2
recipe_agb_s2 <-recipe(formula=AGB.Chave ~., data = rr_sf) %>%
  step_filter(is.na(AGB.Chave) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","Height.q95","HH_SAO","HV_SAO","VH_SAO","VV_SAO",
                   "HHHV_SAO","sqrtHHHV_SAO","HHHV_Ratio_SAO",
                   "VHVV_SAO","sqrtVHVV_SAO","VHVV_Ratio_SAO",
                   "Entropy_HHHV_SAO","Anisotropy_HHHV_SAO","Alpha_HHHV_SAO",
                   "Entropy_VHVV_SAO","Anisotropy_VHVV_SAO","Alpha_VHVV_SAO",
                   "Coherence_HV_SAO","Coherence_HH_SAO",
                   "Coherence_VH_SAO","Coherence_VV_SAO",
                   "Coherence_VH_S1","Coherence_VV_S1",
                   "Entropy_VHVV_S1","Anisotropy_VHVV_S1","Alpha_VHVV_S1","VH_S1","VV_S1",
                   "VHVV_Ratio_S1","VHVV_S1","sqrtVHVV_S1","Elevation_Cop30m"))) 

recipe_ch_s2 <-recipe(formula=Height.q95 ~., data = rr_sf) %>%
  step_filter(is.na(Height.q95) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","AGB.Chave","HH_SAO","HV_SAO","VH_SAO","VV_SAO",
                   "HHHV_SAO","sqrtHHHV_SAO","HHHV_Ratio_SAO",
                   "VHVV_SAO","sqrtVHVV_SAO","VHVV_Ratio_SAO",
                   "Entropy_HHHV_SAO","Anisotropy_HHHV_SAO","Alpha_HHHV_SAO",
                   "Entropy_VHVV_SAO","Anisotropy_VHVV_SAO","Alpha_VHVV_SAO",
                   "Coherence_HV_SAO","Coherence_HH_SAO",
                   "Coherence_VH_SAO","Coherence_VV_SAO",
                   "Coherence_VH_S1","Coherence_VV_S1",
                   "Entropy_VHVV_S1","Anisotropy_VHVV_S1","Alpha_VHVV_S1","VH_S1","VV_S1",
                   "VHVV_Ratio_S1","VHVV_S1","sqrtVHVV_S1","Elevation_Cop30m")))  

# Recipe using only Sentinel-1
recipe_agb_s1 <-recipe(formula=AGB.Chave ~., data = rr_sf) %>%
  step_filter(is.na(AGB.Chave) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","Height.q95","Blue_S2","Green_S2","Red_S2","RedEdge1_S2","RedEdge2_S2",
                   "RedEdge3_S2","NIR1_S2","RedEdge4_S2","SWIR1_S2","SWIR2_S2",
                   "AFRI1600_S2","BI_S2","BITM_S2","BIXS_S2","DBSI_S2","IKAW_S2","LSWI_S2","MBI_S2",      
                   "MRBVI_S2","MSI_S2","NDII_S2","NDMI_S2","NDVI_S2",'MTCI_S2','REDSI_S2', 'BI_S2', 
                   "PSRI_S2","RCC_S2","RENDVI_S2","SAVI_S2","WFI_S2","MDI_S2",
                   "HH_SAO","HV_SAO","VH_SAO","VV_SAO",
                   "HHHV_SAO","sqrtHHHV_SAO","HHHV_Ratio_SAO",
                   "VHVV_SAO","sqrtVHVV_SAO","VHVV_Ratio_SAO",
                   "Entropy_HHHV_SAO","Anisotropy_HHHV_SAO","Alpha_HHHV_SAO",
                   "Entropy_VHVV_SAO","Anisotropy_VHVV_SAO","Alpha_VHVV_SAO",
                   "Coherence_HV_SAO","Coherence_HH_SAO",
                   "Coherence_VH_SAO","Coherence_VV_SAO",
                   "Elevation_Cop30m")))  


recipe_ch_s1 <-recipe(formula=Height.q95 ~., data = rr_sf) %>%
  step_filter(is.na(Height.q95) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","AGB.Chave","Blue_S2","Green_S2","Red_S2","RedEdge1_S2","RedEdge2_S2",
                   "RedEdge3_S2","NIR1_S2","RedEdge4_S2","SWIR1_S2","SWIR2_S2",
                   "AFRI1600_S2","BI_S2","BITM_S2","BIXS_S2","DBSI_S2","IKAW_S2","LSWI_S2","MBI_S2",      
                   "MRBVI_S2","MSI_S2","NDII_S2","NDMI_S2","NDVI_S2",'MTCI_S2','REDSI_S2', 'BI_S2', 
                   "PSRI_S2","RCC_S2","RENDVI_S2","SAVI_S2","WFI_S2","MDI_S2",
                   "HH_SAO","HV_SAO","VH_SAO","VV_SAO",
                   "HHHV_SAO","sqrtHHHV_SAO","HHHV_Ratio_SAO",
                   "VHVV_SAO","sqrtVHVV_SAO","VHVV_Ratio_SAO",
                   "Entropy_HHHV_SAO","Anisotropy_HHHV_SAO","Alpha_HHHV_SAO",
                   "Entropy_VHVV_SAO","Anisotropy_VHVV_SAO","Alpha_VHVV_SAO",
                   "Coherence_HV_SAO","Coherence_HH_SAO",
                   "Coherence_VH_SAO","Coherence_VV_SAO",
                   "Elevation_Cop30m"))) 

# Recipe using only SAOCOM
recipe_agb_sao <-recipe(formula=AGB.Chave ~., data = rr_sf) %>%
  step_filter(is.na(AGB.Chave) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","Height.q95","Blue_S2","Green_S2","Red_S2","RedEdge1_S2","RedEdge2_S2",
                   "RedEdge3_S2","NIR1_S2","RedEdge4_S2","SWIR1_S2","SWIR2_S2",
                   "AFRI1600_S2","BI_S2","BITM_S2","BIXS_S2","DBSI_S2","IKAW_S2","LSWI_S2","MBI_S2",      
                   "MRBVI_S2","MSI_S2","NDII_S2","NDMI_S2","NDVI_S2",'MTCI_S2','REDSI_S2', 'BI_S2', 
                   "PSRI_S2","RCC_S2","RENDVI_S2","SAVI_S2","WFI_S2","MDI_S2",
                   "Coherence_VH_S1","Coherence_VV_S1",
                   "Entropy_VHVV_S1","Anisotropy_VHVV_S1","Alpha_VHVV_S1","VH_S1","VV_S1",
                   "VHVV_Ratio_S1","VHVV_S1","sqrtVHVV_S1","Elevation_Cop30m")))  


recipe_ch_sao <-recipe(formula=Height.q95 ~., data = rr_sf) %>%
  step_filter(is.na(Height.q95) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","AGB.Chave","Blue_S2","Green_S2","Red_S2","RedEdge1_S2","RedEdge2_S2",
                   "RedEdge3_S2","NIR1_S2","RedEdge4_S2","SWIR1_S2","SWIR2_S2",
                   "AFRI1600_S2","BI_S2","BITM_S2","BIXS_S2","DBSI_S2","IKAW_S2","LSWI_S2","MBI_S2",      
                   "MRBVI_S2","MSI_S2","NDII_S2","NDMI_S2","NDVI_S2",'MTCI_S2','REDSI_S2', 'BI_S2', 
                   "PSRI_S2","RCC_S2","RENDVI_S2","SAVI_S2","WFI_S2","MDI_S2",
                   "Coherence_VH_S1","Coherence_VV_S1",
                   "Entropy_VHVV_S1","Anisotropy_VHVV_S1","Alpha_VHVV_S1","VH_S1","VV_S1",
                   "VHVV_Ratio_S1","VHVV_S1","sqrtVHVV_S1","Elevation_Cop30m")))  

# Recipe using only Sentinel-2 and SAOCOM
recipe_agb_s2sao <-recipe(formula=AGB.Chave ~., data = rr_sf) %>%
  step_filter(is.na(AGB.Chave) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","Height.q95",
                   "Coherence_VH_S1","Coherence_VV_S1",
                   "Entropy_VHVV_S1","Anisotropy_VHVV_S1","Alpha_VHVV_S1","VH_S1","VV_S1",
                   "VHVV_Ratio_S1","VHVV_S1","sqrtVHVV_S1","Elevation_Cop30m")))  


recipe_ch_s2sao <-recipe(formula=Height.q95 ~., data = rr_sf) %>%
  step_filter(is.na(Height.q95) == FALSE) %>%  
  step_rm(any_of(c("x", "y", "geometry","AGB.Chave",
                   "Coherence_VH_S1","Coherence_VV_S1",
                   "Entropy_VHVV_S1","Anisotropy_VHVV_S1","Alpha_VHVV_S1","VH_S1","VV_S1",
                   "VHVV_Ratio_S1","VHVV_S1","sqrtVHVV_S1","Elevation_Cop30m")))  

# # Recipe using only SAOCOM and Sentinel-1 (SAR)
recipe_agb_sar <-recipe(formula=AGB.Chave ~., data = rr_sf) %>%
  step_filter(is.na(AGB.Chave) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","Height.q95","Blue_S2","Green_S2","Red_S2","RedEdge1_S2","RedEdge2_S2",
                   "RedEdge3_S2","NIR1_S2","RedEdge4_S2","SWIR1_S2","SWIR2_S2",
                   "AFRI1600_S2","BI_S2","BITM_S2","BIXS_S2","DBSI_S2","IKAW_S2","LSWI_S2","MBI_S2",      
                   "MRBVI_S2","MSI_S2","NDII_S2","NDMI_S2","NDVI_S2", 'MTCI_S2','REDSI_S2', 'BI_S2',  
                   "PSRI_S2","RCC_S2","RENDVI_S2","mND705_S2","SAVI_S2","WFI_S2","MDI_S2","Elevation_Cop30m"))) 


recipe_ch_sar <-recipe(formula= Height.q95~., data = rr_sf) %>%
  step_filter(is.na(Height.q95) == FALSE) %>% 
  step_rm(any_of(c("x", "y", "geometry","AGB.Chave","Blue_S2","Green_S2","Red_S2","RedEdge1_S2","RedEdge2_S2",
                   "RedEdge3_S2","NIR1_S2","RedEdge4_S2","SWIR1_S2","SWIR2_S2",
                   "AFRI1600_S2","BI_S2","BITM_S2","BIXS_S2","DBSI_S2","IKAW_S2","LSWI_S2","MBI_S2",      
                   "MRBVI_S2","MSI_S2","NDII_S2","NDMI_S2","NDVI_S2",  
                   "PSRI_S2","RCC_S2","RENDVI_S2","mND705_S2","SAVI_S2","WFI_S2","MDI_S2","Elevation_Cop30m"))) 


#####################  MODEL TRAINING  ######################
# Create lists of preprocessing recipes
preprocessing_agb <-  list(All = recipe_agb, 
                            Sentinel2 = recipe_agb_s2, 
                            Sentinel1 = recipe_agb_s1, 
                            SAOCOM = recipe_agb_sao,
                            Sentinel1_2 = recipe_agb_sentinel, 
                            Sentinel2_SAOCOM = recipe_agb_s2sao,
                            Sentinel1_SAOCOM = recipe_agb_sar,
                            Elevation = recipe_agb_elev)

preprocessing_ch <-  list(All= recipe_ch,
                          Sentinel2 = recipe_ch_s2, 
                          Sentinel1 = recipe_ch_s1, 
                          SAOCOM = recipe_ch_sao,
                          Sentinel1_2 = recipe_ch_sentinel, 
                          Sentinel2_SAOCOM = recipe_ch_s2sao,
                          Sentinel1_SAOCOM = recipe_ch_sar,
                          Elevation = recipe_ch_elev)

# Create workflow sets combining recipes with models
rf_wf_ch <- workflow_set(preproc = preprocessing_ch, 
                        models = list(model=rf_model_ch), 
                        cross = FALSE)

rf_wf_agb <- workflow_set(preproc = preprocessing_agb, 
                        models = list(model=rf_model_agb), 
                        cross = FALSE)

# Define evaluation metrics and control settings
list_metrics <- metric_set(rmse,rsq)
ctrl <- control_resamples(save_workflow = TRUE,save_pred = TRUE)

# Train models using spatial cross-validation
rf_results_ch <- rf_wf_ch %>% 
  workflow_map("fit_resamples", metrics = list_metrics,
               seed =100, 
               control = ctrl,  
               resamples =spatial_folds_train21) 

rf_results_agb <- rf_wf_agb %>% 
   workflow_map("fit_resamples", metrics = list_metrics,
               seed =100, 
               control = ctrl,
               resamples = spatial_folds_train21) 

# Extract best performing models
fitbest_train_ch <- fit_best(rf_results_ch)
fitbest_train_agb <- fit_best(rf_results_agb)


###################  VARIABLE IMPORTANCE  ######################

# Extract and calculate variable importance as percentage
importance_allvar_ch <- extract_fit_parsnip(fitbest_train_ch) %>% 
  vip::vi()%>%
  mutate(IP = 100 * Importance / sum(Importance))
importance_allvar_ch %>% #write.csv("your/path/Importance_AllVar_ch.csv")

importance_allvar_agb <- extract_fit_parsnip(fitbest_train_agb) %>% 
  vip::vi()%>%
  mutate(IP = 100 * Importance / sum(Importance))
importance_allvar_agb %>% #write.csv("your/path/importance_allvar_agb.csv")

# Function to extract data source from variable names
extract_suffix <- function(x) {
  sapply(strsplit(as.character(x), "_"), function(y) tail(y, 1))
}

# Combine importance results
importance_all <- bind_rows(
  importance_allvar_ch %>% mutate(Source  = extract_suffix(Variable), Type= "CH"),
  importance_allvar_agb %>% mutate(Source= extract_suffix(Variable), Type  = "AGB")
) %>%
  select(Variable,  Source,Type, everything())


############### FEATURE SELECTION AND MULTICOLLINEARITY ###################

# Feature selection
# Extract the names of the most & least important variables from the 'Variable' column
first_vars_ch <- head(importance_allvar_ch$Variable, 10) 
first_vars_agb <- head(importance_allvar_agb$Variable, 10) 

last_vars_ch <- tail(importance_allvar_ch$Variable, 10) 
last_vars_agb <- tail(importance_allvar_agb$Variable, 10) 

# Filter multicollinear variables through Variance Inflation Factor (VIF)
# Remove low importance variables and filter for multicollinearity
rr_modelbp_select_ch <- subset(rr_modelbp, setdiff(names(rr_modelbp), last_vars_ch))        
rr_modelbp_select_agb <- subset(rr_modelbp, setdiff(names(rr_modelbp),last_vars_agb))

# Filter collinear variables using VIF
rr_filter_ch <- filter_collinear( 
  rr_modelbp_select_ch,
  na.rm=TRUE,
  cutoff = 5, 
  verbose = TRUE,
  names = TRUE,
  to_keep = c(first_vars_ch, 'Height.q95'),
  method = "vif_step",
  max_cells = Inf,
  exhaustive = FALSE)

rr_filter_agb <- filter_collinear( 
  rr_modelbp_select_agb,
  na.rm=TRUE,
  cutoff = 5, 
  verbose = TRUE,
  names = TRUE,
  to_keep = c(first_vars_agb, 'AGB.Chave'),
  method = "vif_step",
  max_cells = Inf,
  exhaustive = FALSE)


###################### HYPERPARAMETER TUNING  ########################
# Note: this part may require a substantial computational time

# Prepare data for hyperparameter tuning
test_data21_ch <- test_data21 %>% select(any_of(rr_filter_ch), -AGB.Chave) %>% 
  drop_na(Height.q95) %>% st_as_sf(coords = c("x", "y"), crs = 32750)#%>% st_drop_geometry()
test_data21_agb <- test_data21 %>% select(any_of(rr_filter_agb), -Height.q95)%>% 
  drop_na(AGB.Chave) %>% st_as_sf(coords = c("x", "y"), crs = 32750)#%>% st_drop_geometry()

# Convert to data frames without geometry
test21_sf_ch <- test_data21_ch %>%  st_drop_geometry()
test21_sf_agb <- test_data21_agb %>%  st_drop_geometry()

# Create spatial folds for tuning
spatial_folds_test21_ch <- test_data21_ch  %>% 
  spatial_block_cv(v=10, buffer=500,method="random") 
spatial_folds_test21_agb <- test_data21_agb %>% 
  spatial_block_cv(v=10, buffer=500, method="random") 

# Define recipes for models for tuning 
rf_recipe_ch <- recipe(Height.q95~ ., data = test21_sf_ch, num.threads = 100) %>% 
  step_filter(is.na(Height.q95) == FALSE) 

rf_recipe_agb <- recipe(AGB.Chave ~ ., data = test21_sf_agb, num.threads = 100) %>% 
  step_filter(is.na(AGB.Chave) == FALSE)  

# Define tunable hyperparameters in random forest model: 
# find the best mtry, number of trees and min_n
rf_model_ch_tune <- 
  rand_forest (mtry = tune(), trees = tune(), min_n = tune())%>%
  set_mode("regression") %>%
  set_engine("ranger",num.threads = 100, 
             keep.inbag = TRUE,    
             importance = "permutation",
             seed = 100) 

rf_model_agb_tune <- 
  rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_mode("regression") %>%
  set_engine("ranger",num.threads = 100, 
             keep.inbag = TRUE,     
             importance = "permutation",
             seed = 100) 
                 
# Create workflows for tuning
rf_workflow_ch <- workflow() %>%
  add_recipe(rf_recipe_ch) %>%
  add_model(rf_model_ch_tune)

rf_workflow_agb <- workflow() %>%
  add_recipe(rf_recipe_agb) %>%
  add_model(rf_model_agb_tune)

###################### TUNING PROCESS ######################
# Define tuning grid
tune_grid <- grid_regular(
  mtry(range = c(5, 20)),  
  min_n(range = c(5, 20)),
  trees(range = c(1500, 2000)),levels=5)

# Perform hyperparameter tuning
tune_results_ch <- tune_grid(
  rf_workflow_ch,
  resamples = spatial_folds_test21_ch, 
  control = control_grid(save_workflow = TRUE, save_pred = TRUE),
  grid = tune_grid, 
  metrics = metric_set(rsq, rmse))


tune_results_agb <- tune_grid(
  rf_workflow_agb,
  resamples = spatial_folds_test21_agb, 
  control = control_grid(save_workflow = TRUE, save_pred = TRUE),
  grid = tune_grid, 
  metrics = metric_set(rsq, rmse))

# Select best hyperparameters
best_params_ch <- select_best(tune_results_ch, metric=c("rmse")) 
best_params_agb <- select_best(tune_results_agb, metric=c("rmse")) 


############ MODEL PERFORMANCE WITH ADDITIONAL REDUCED VARIABLES ##############

# Adjust hyperparameters after tuning: 
# rand_forest(mtry = ..., trees = ..., min_n = ...)

# Define tuned random forest model with optimized hyperparameters
rf_model_ch_tuned <- rand_forest(mtry = ..., trees = ..., min_n = ...)  %>%
  set_engine("ranger",
             keep.inbag = TRUE,  
             importance = "permutation",
             seed = 100) %>%
  set_mode("regression") 

rf_model_agb_tuned <- rand_forest(mtry = ..., trees = ..., min_n = ...)  %>%
  set_engine("ranger",
             keep.inbag = TRUE,    
             importance = "permutation",
             seed = 100) %>%
  set_mode("regression") 

# Define recipes using reduced variable sets after feature selection
recipe_ch_reduced <- recipe(formula=Height.q95 ~., data =rr_sf) %>%
  step_filter(is.na(Height.q95) == FALSE) %>% 
  step_rm(-any_of(rr_filter_ch)) %>%
  step_rm(any_of(c("x", "y", "geometry","AGB.Chave")))

recipe_agb_reduced <- recipe(formula=AGB.Chave ~., data = rr_sf) %>% 
  step_filter(is.na(AGB.Chave) == FALSE) %>% 
  step_rm(-any_of(rr_filter_agb)) %>% 
  step_rm(any_of(c("x", "y","geometry","Height.q95")))  


# Create list of preprocessing recipes including reduced variable set
preprocessing_ch_reduced <-   list(All = recipe_ch,
                                  Reduced = recipe_ch_reduced,
                                  Sentinel2 = recipe_ch_s2, 
                                  Sentinel1 = recipe_ch_s1, 
                                  SAOCOM = recipe_ch_sao,
                                  Sentinel1_2 = recipe_ch_sentinel, 
                                  Sentinel2_SAOCOM = recipe_ch_s2sao,
                                  Sentinel1_SAOCOM = recipe_ch_sar,
                                  Elevation = recipe_ch_elev)

preprocessing_agb_reduced <-   list(All = recipe_agb, 
                                    Reduced = recipe_agb_reduced,
                                    Sentinel2 = recipe_agb_s2, 
                                    Sentinel1 = recipe_agb_s1, 
                                    SAOCOM = recipe_agb_sao,
                                    Sentinel1_2 = recipe_agb_sentinel, 
                                    Sentinel2_SAOCOM = recipe_agb_s2sao,
                                    Sentinel1_SAOCOM = recipe_agb_sar,
                                    Elevation = recipe_agb_elev)

# Create workflow set with reduced variables
rf_wf_ch_reduced <- workflow_set(preproc = preprocessing_ch_reduced, 
                                models = list(model=rf_model_ch_tuned), 
                                cross = FALSE)

rf_wf_agb_reduced <- workflow_set(preproc = preprocessing_agb_reduced, 
                                models = list(model=rf_model_agb_tuned), 
                                cross = FALSE)

# Train models with reduced variables using spatial cross-validation
rf_results_ch_reduced <- 
  rf_wf_ch_reduced %>% 
  workflow_map("fit_resamples", metrics = list_metrics,
               seed =100,
               control = ctrl,  
               resamples =spatial_folds_train21) 

rf_results_agb_reduced <- 
  rf_wf_agb_reduced %>% 
  workflow_map("fit_resamples", metrics = list_metrics,
               seed =100, 
               control = ctrl,
               resamples = spatial_folds_train21) 

# Extract the best performing model from reduced variable results
fitbest_train_ch_reduced <- fit_best(rf_results_ch_reduced)
fitbest_train_agb_reduced <- fit_best(rf_results_agb_reduced)


# Graphs - Create visualizations for model performance

# Collect and display performance metrics 
rf_results_ch_reduced %>%
  collect_metrics() 
rf_results_agb_reduced %>%
  collect_metrics() 

theme_set(theme_bw())
conflicted::conflicts_prefer(dplyr::mutate)


# Model performance graph
mph <- rf_results_ch_reduced %>%
  collect_metrics() %>%
  mutate(wflow_id = fct_reorder(wflow_id, mean),
         .metric = case_when(
           .metric == "rmse" ~ "RMSE",
           .metric == "rsq" ~ "R²",
           TRUE ~ .metric 
         )) %>%
  ggplot(aes(y=wflow_id, x=mean,color = wflow_id)) +
  geom_errorbar(width=.1, aes(xmin=mean-std_err, xmax=mean+std_err)) +
  geom_point(size = 2) +scale_color_viridis(discrete = TRUE, option = "D") +
  facet_grid(scales = "free", switch = "y",~fct_relevel(.metric,"R²","RMSE")) +
  theme_bw()+
  labs(x="", y="")+
  theme(legend.position = "none")+ 
  ggtitle('A. Canopy height')

mpa <- rf_results_agb_reduced %>%
  collect_metrics() %>%
  mutate(wflow_id = fct_reorder(wflow_id, mean),
         .metric = case_when(
           .metric == "rmse" ~ "RMSE",
           .metric == "rsq" ~ "R²",
           TRUE ~ .metric  
         )) %>%
  ggplot(aes(y = wflow_id, x = mean, color = wflow_id)) +
  geom_errorbar(width = 0.1, aes(xmin = mean - std_err, xmax = mean + std_err)) +
  geom_point(size = 2) +
  scale_color_viridis(discrete = TRUE, option = "D") +
  facet_grid(scales = "free", switch = "y",~fct_relevel(.metric,"R²","RMSE" )) + 
  theme_bw() +
  labs(x = "", y = "") +
  theme(legend.position = "none") +
  ggtitle('B. Aboveground biomass')


mph/mpa
#ggsave("your/path/model_performance_final_tuned_reduced.tiff",  width = 6.5, height = 5.5, dpi = 300, compression = "lzw")


# Prediction graph - Create observed vs predicted scatter plots
# Get predictions and metrics from resampling
preds_ch <- rf_results_ch_reduced %>% collect_predictions() %>% 
  filter(wflow_id %in% c("All_model", "Reduced_model"))  
   
preds_agb <- rf_results_agb_reduced %>% collect_predictions() %>% 
 filter(wflow_id %in% c("All_model", "Reduced_model"))  

# Calculate performance metrics
metrics_ch <- rf_results_ch_reduced %>% 
  collect_metrics() %>%
  filter(.metric %in% c("rmse", "rsq")) %>%
  group_by(wflow_id, .metric) %>% 
  dplyr::summarize(mean_value = mean(mean, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = .metric, values_from = mean_value) %>%
  mutate(label = sprintf("RMSE = %.1f\nR² = %.2f", rmse, rsq)) %>% 
  filter(wflow_id %in% c("All_model", "Reduced_model"))

metrics_agb <- rf_results_agb_reduced %>% 
  collect_metrics() %>%
  filter(.metric %in% c("rmse", "rsq")) %>%
  group_by(wflow_id, .metric) %>% 
  dplyr::summarize(mean_value = mean(mean, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = .metric, values_from = mean_value) %>%
  mutate(label = sprintf("RMSE = %.1f\nR² = %.2f", rmse, rsq)) %>% 
  filter(wflow_id %in% c("All_model", "Reduced_model"))

#Reorder workflows by response variable values for better visualization
preds_ch <- preds_ch %>% mutate(wflow_id = fct_reorder(wflow_id, Height.q95))
preds_agb <- preds_agb %>% mutate(wflow_id = fct_reorder(wflow_id, AGB.Chave))

# Combine predictions with metrics for plotting
preds_metrics_ch <- preds_ch %>%  left_join(metrics_ch, by = "wflow_id")
preds_metrics_agb <- preds_agb %>%  left_join(metrics_agb, by = "wflow_id")

# Create observed vs predicted plot
all_red_ch_plot <- ggplot(preds_metrics_ch , aes(x =  Height.q95, y = .pred)) +
  geom_pointdensity(show.legend = FALSE, size = 1, alpha = 0.6) +
  scale_color_viridis_c() +
  geom_abline(col = "black", lty = 3) +
  geom_label(
    data = metrics_ch,
    aes(x = 1, y = 20, label = label),
    hjust = 0, vjust = 1, size = 3,
    fill = "white", alpha = 0.7,
    inherit.aes = FALSE)+
  geom_smooth(col='blue',method=lm, se=TRUE, fullrange=TRUE)+
  facet_grid(~ wflow_id) +
  coord_cartesian(xlim = c(0, 22), ylim = c(0, 22))+
  labs(y='Predicted canopy height (m)', x='Observed canopy height (m)', title='A. Canopy height')+
  theme_bw()

all_red_agb_plot <- ggplot(preds_metrics_agb, aes(x = AGB.Chave, y = .pred)) +
  geom_pointdensity(show.legend = FALSE, size = 1, alpha = 0.6) +
  scale_color_viridis_c() +
  geom_abline(col = "black", lty = 3) +
  geom_label(
    data = metrics_agb,
    aes(x = 20, y = 400, label = label),
    hjust = 0, vjust = 1, size = 3,
    fill = "white", alpha = 0.7,
    inherit.aes = FALSE)+
  geom_smooth(col='blue',method=lm, se=TRUE, fullrange=TRUE)+
  facet_grid(~ wflow_id) +
  coord_cartesian(xlim = c(0, 410), ylim = c(0, 410))+
  labs(x=expression("Observed aboveground biomass (Mg ha"^-1 * ")"), 
       y= expression("Predicted aboveground biomass (Mg ha"^-1 * ")"), title='B. Aboveground biomass' )+
  theme_bw()

all_red_ch_plot/all_red_agb_plot
#ggsave("your/path/Prediction_CH_AGB_final_tuned_reduced.tiff",  width = 6.5, height = 6, dpi = 800, compression = "lzw")

# Metrics calculation: Bias and Relative RMSE (%)
## RRMSE CH
(metrics_ch$rmse) /(max(train21_sf$Height.q95)-min(train21_sf$Height.q95))*100 

## RRMSE AGB 
(metrics_agb$rmse)/(max(train21_sf$AGB.Chave)-min(train21_sf$AGB.Chave))*100 

## Bias CH
preds_ch_all = preds_ch %>% filter(wflow_id=='All_model')
(sum(preds_ch_all$Height.q95)-sum(preds_ch_all$.pred))/max(preds_ch_all$.row)

preds_ch_reduced = preds_ch %>% filter(wflow_id=='Reduced_model')
(sum(preds_ch_reduced$Height.q95)-sum(preds_ch_reduced$.pred))/max(preds_ch_reduced$.row)

## Bias AGB
preds_agb_all = preds_a%>% filter(wflow_id=='All_model')
(sum(preds_agb_all$AGB.Chave)-sum(preds_agb_all$.pred))/max(preds_agb_all$.row)

preds_agb_reduced = preds_agb %>% filter(wflow_id=='Reduced_model')
(sum(preds_agb_reduced$AGB.Chave)-sum(preds_agb_reduced$.pred))/max(preds_agb_reduced$.row)



#############  FINAL MODEL PRODUCTION  ##############

# Create spatial split for final model training 
set.seed(100)
rr_block_split22ch <-
  spatial_initial_split(
    data=st_as_sf(train_data1),
    prop = 1/4, 
    buffer=45,
    strategy = spatial_block_cv)

set.seed(100)
rr_block_split22agb <-
  spatial_initial_split(
    data=st_as_sf(train_data1),
    prop = 1/4, 
    buffer=45,
    strategy = spatial_block_cv)

# Extract training and testing data for model finalization
train_data22ch = training(rr_block_split22ch) %>% drop_na(Height.q95) %>% 
  st_as_sf(coords = c("x", "y"), crs = 32750)
test_data22ch = testing(rr_block_split22ch) %>% drop_na(Height.q95) %>% 
  st_as_sf(coords = c("x", "y"), crs = 32750)

train_data22agb = training(rr_block_split22agb) %>% drop_na(AGB.Chave) %>% 
  st_as_sf(coords = c("x", "y"), crs = 32750)
test_data22agb = testing(rr_block_split22agb) %>% drop_na(AGB.Chave) %>% 
  st_as_sf(coords = c("x", "y"), crs = 32750)

set.seed(100)
spatial_folds_train22_ch <- train_data22ch %>% spatial_block_cv(v=10, repeats=10, buffer=500,method="random") 

set.seed(100)
spatial_folds_train22_agb <- train_data22agb %>% spatial_block_cv(v=10, repeats=10, buffer=500, method="random")


# Extract predictions from the best hyperparameter configuration 

final_model_preds_ch <- tune_results_ch %>%
  collect_predictions()%>%
  inner_join(show_best(tune_results_ch, metric = "rmse", n = 1) %>% 
               select(.config),by = ".config")
final_model_preds_ch 

final_model_preds_agb  <- tune_results_agb %>%
  collect_predictions()%>%
  inner_join(
    show_best(tune_results_agb, metric = "rmse", n = 1) %>% select(.config),
    by = ".config")
final_model_preds_agb


tune_results_ch %>%
  collect_metrics()

tune_results_agb %>%
  collect_metrics()

fit_train_tune_ch <- fit_best(tune_results_ch)
fit_train_tune_agb <- fit_best(tune_results_agb)

# Finalize workflow with best parameters and train on full training data
last_rf_workflow_ch <- rf_workflow_ch %>% update_model(rf_model_ch_tuned)
last_rf_workflow_agb <- rf_workflow_agb %>% update_model(rf_model_agb_tuned)


rf_spatial_ch <- tune::fit_resamples(last_rf_workflow_ch,
                                    resamples = spatial_folds_train22_ch,
                                    control = control_resamples(save_pred = TRUE,save_workflow = TRUE))

rf_spatial_agb <- tune::fit_resamples(last_rf_workflow_agb,
                                    resamples = spatial_folds_train22_agb,
                                    control = control_resamples(save_pred = TRUE,save_workflow = TRUE))

final_model_ch <- fit_best(rf_spatial_ch)
final_model_agb <- fit_best(rf_spatial_agb)

# the last fit
set.seed(100)
last_rf_fit_ch <-  last_rf_workflow_ch %>%  last_fit(rr_block_split22ch)
set.seed(100)
last_rf_fit_agb <-  last_rf_workflow_agb %>%  last_fit(rr_block_split22agb)

# performance assessment on the test data
# Predict on test set
final_test_ch <- predict(final_model_ch, new_data = test22_sf_ch) %>%  bind_cols(test22_sf_ch) 
final_test_agb <- predict(final_model_agb, new_data = test22_sf_agb) %>%  bind_cols(test22_sf_agb) 

test_performance_ch <- final_test_ch  %>% list_metrics(truth = Height.q95, estimate = .pred)
test_performance_agb <- final_test_agb  %>% list_metrics(truth = AGB.Chave, estimate = .pred)
str(test_performance_ch)

# Save final model
#saveRDS(final_model_ch, "your/path/final_rf_ch_model.rds")
#saveRDS(final_model_agb, "your/path/final_rf_agb_model.rds")

## RRMSE CH - TEST 
max(test_performance_ch$.estimate) /(max(test22_sf_ch$Height.q95)-min(test22_sf_ch$Height.q95))*100 
## AGB - TEST
max(test_performance_agb$.estimate)/(max(test22_sf_agb$AGB.Chave)-min(test22_sf_agb$AGB.Chave))*100


# Importance from trained dataset

# Extract final importance graph 
importance_df_ch <- extract_fit_parsnip(final_model_ch) %>% 
  vip::vi()%>%
  mutate(IP = 100 * Importance / sum(Importance))

importance_df_agb <- extract_fit_parsnip(final_model_agb) %>% 
  vip::vi()%>%
  mutate(IP = 100 * Importance / sum(Importance))

importance_final <- bind_rows(
  importance_df_ch %>% mutate(Source  = extract_suffix(Variable), Type= "CH"),
  importance_df_agb %>% mutate(Source= extract_suffix(Variable), Type  = "AGB")) %>%
  select(Variable,  Source,Type, everything())
#write.csv(importance_final, "your/path/Importance_Final_combined.csv")

# Variable Importance graph
# Note: Rename several variables for ease of understanding
imp_ch <- importance_final %>% mutate(across(c(Type,Source), as.factor)) %>%
  filter(Type=='CH') %>%
  mutate(Variable = recode(Variable,
                           "HHHV_SAO" = "HHxHV_SAO",
                           "sqrtHHHV_SAO" = "√HHxHV_SAO",
                           "VHVV_S1" = "VHxVV_S1",
                           "VHVV_SAO" = "VHxVV_SAO",
                           "VHVV_Ratio_SAO" = "VH/VV_Ratio_SAO")) %>%
  mutate(name = fct_reorder(Variable, IP)) %>%
  ggplot(aes(x=name, y=IP, fill=Source)) +
  geom_bar(stat="identity", position=position_dodge(), col='darkgrey')+
  theme_minimal() +coord_flip()+
  scale_fill_brewer(palette="RdYlBu", limits = c("Cop30m","S1", "S2", "SAO"),   # Order option='A',alpha=0.6, 
                    labels = c("Copernicus DEM","Sentinel-1", "Sentinel-2", "SAOCOM"))+ 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
        legend.position = "none",axis.text = element_text(color="black")) +ylim(0,16)+
  labs(title="A. Canopy height", y="Permutation Importance (%)", x="Predictor variables") 


imp_agb <- importance_final %>% mutate(across(c(Type,Source), as.factor)) %>%
  filter(Type=='AGB') %>%
  mutate(Variable = recode(Variable,
                           "HHHV_SAO" = "HHxHV_SAO",
                           "sqrtHHHV_SAO" = "√HHxHV_SAO",
                           "VHVV_S1" = "VHxVV_S1",
                           "VHVV_SAO" = "VHxVV_SAO",
                           "VHVV_Ratio_SAO" = "VH/VV_Ratio_SAO")) %>%
  mutate(name = fct_reorder(Variable, IP)) %>%
  ggplot(aes(x=name, y=IP, fill=Source)) +
  geom_bar(stat="identity", position=position_dodge(), col='darkgrey')+
  theme_minimal() +coord_flip()+
  scale_fill_brewer(palette="RdYlBu", limits = c("Cop30m","S1", "S2", "SAO"), 
                    labels = c("Copernicus DEM","Sentinel-1", "Sentinel-2", "SAOCOM"))+ 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
        axis.text = element_text(color="black")) +ylim(0,16)+
  labs(title="B. Aboveground biomass", y="Permutation Importance (%)", x="") 

imp_ch + imp_agb 
#ggsave("your/path/VIP_Height_AGB_FINAL_TUNED_PUBS_reduced.tiff", width = 8, height =5, dpi = 400, compression = "lzw")

################ PREDICTION PLOT #####################

# Get predictions and metrics from resampling
preds_ch_reduced <- last_rf_fit_ch %>% collect_predictions()
preds_agb_reduced <- last_rf_fit_agb %>% collect_predictions() 


metrics_ch_reduced <- last_rf_fit_ch  %>% 
  collect_metrics() %>% 
  group_by(.metric) %>% 
  dplyr::summarize(mean_value = mean( .estimate, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = .metric, values_from = mean_value) %>%
  mutate(label = sprintf("RMSE = %.1f\nR² = %.2f", rmse, rsq)) 

metrics_agb_reduced <- last_rf_fit_agb %>% 
  collect_metrics() %>%
  group_by(.metric) %>% 
  dplyr::summarize(mean_value = mean(.estimate, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = .metric, values_from = mean_value) %>%
  mutate(label = sprintf("RMSE = %.1f\nR² = %.2f", rmse, rsq)) 

## Bias-CH
(sum(preds_ch_reduced$Height.q95)-sum(preds_ch_reduced$.pred))/nrow(preds_ch_reduced)

## Bias-AGB
(sum(preds_agb_reduced$AGB.Chave)-sum(preds_agb_reduced$.pred))/nrow(preds_agb_reduced)
(sum(preds_agb_reduced$AGB.Chave-preds_agb_reduced$.pred))/nrow(preds_agb_reduced)


final_red_ch_plot <- ggplot(preds_ch_reduced, aes(y =Height.q95, x = .pred)) +
  geom_pointdensity(show.legend = TRUE, size = 1) +
  scale_color_viridis_c() +
  geom_abline(col = "black", lty = 3) +
  geom_label(
    data = metrics_ch_reduced,
    aes(x = 1, y = 21, label = label),
    hjust = 0, vjust = 1, size = 3,
    fill = "white", alpha = 0.7,
    inherit.aes = FALSE)+
  geom_smooth(col='blue',method=lm, se=TRUE, fullrange=TRUE)+
  #facet_grid(~ wflow_id) +
  coord_cartesian(xlim = c(0, 22), ylim = c(0, 22))+
  labs(x='Predicted canopy height (m)', y='Observed canopy height (m)',  col='Number of \npoints',title='A. Canopy height')+
  theme_bw()

final_red_agb_plot <- ggplot(preds_agb_reduced, aes(y = AGB.Chave, x = .pred)) +
  geom_pointdensity(show.legend = TRUE, size = 1) +
  scale_color_viridis_c(breaks = c(2,4, 6, 8,10)) +
  geom_abline(col = "black", lty = 3) +
  geom_label(
    data = metrics_agb_reduced,
    aes(x = 20, y = 400, label = label),
    hjust = 0, vjust = 1, size = 3,
    fill = "white", alpha = 0.7,
    inherit.aes = FALSE)+
  geom_smooth(col='blue',method=lm, se=TRUE, fullrange=TRUE)+
  #facet_grid(~ wflow_id) +
  coord_cartesian(xlim = c(0, 410), ylim = c(0, 410))+
  labs(y=expression("Observed aboveground biomass (Mg ha"^-1 * ")"),
       x= expression("Predicted aboveground biomass (Mg ha"^-1 * ")"), col='Number of \npoints',title='B. Aboveground biomass' )+
  theme_bw()

final_red_ch_plot+final_red_agb_plot
#ggsave("your/path/CH_AGB_tuned_reduced_test.tiff", width = 9, height = 4, dpi = 800, compression = "lzw")


##################  CREATE PREDICTION MAP  ##################

# Function to predict as vector and convert to raster data 

process_predictions <- function(
    model, 
    new_data_sf, 
    predict_type = "numeric",  # "numeric" or "conf_int"
    level = 0.95,             # Confidence level (if predict_type = "conf_int")
    std_error = TRUE,         # Include std.error layer? (if predict_type = "conf_int")
    crs = "EPSG:32750", 
    res = 15, 
    plot = TRUE
) {
  # Step 1: Generate predictions based on type
  preds <- predict(
    model, 
    new_data = new_data_sf, 
    type = predict_type,
    level = level,            # Only used if predict_type = "conf_int"
    std_error = std_error     # Only used if predict_type = "conf_int"
  )
  
  # Step 2: Bind predictions to SF object and extract coordinates
  eval_data <- new_data_sf %>% 
    bind_cols(preds) %>%
    mutate(
      x = unlist(map(geometry, 1)),  # Extract X coords
      y = unlist(map(geometry, 2))   # Extract Y coords
    )
  
  # Step 3: Rasterize predictions using terra
  r <- terra::rast(eval_data, resolution = res, crs = crs)
  
  # Get column names of predictions (e.g., ".pred", ".std_error", etc.)
  pred_cols <- names(preds)
  
  # Rasterize each prediction column and name layers accordingly
  rasterized <- lapply(pred_cols, function(col) {
    terra::rasterize(
      terra::vect(eval_data), 
      r, 
      field = col, 
      fun = mean
    ) %>% 
      setNames(col)  # Name the layer after the original column
  }) %>% 
    setNames(pred_cols)  # Name list elements after original columns
  
  # Convert to multi-layer SpatRaster if multiple outputs exist
  if (length(rasterized) > 1) {
    rasterized <- terra::rast(rasterized)
  } else {
    rasterized <- rasterized[[1]]  # Single-layer SpatRaster
  }
}

map_bp_ch_num <- process_predictions(final_model_ch, rrbp_sf, predict_type = "numeric")
map_bp_ch_ci <- process_predictions(final_model_ch, rrbp_sf, predict_type = "conf_int", std_error = TRUE)
map_bp_agb_num <- process_predictions(final_model_agb, rrbp_sf, predict_type = "numeric")
map_bp_agb_ci <- process_predictions(final_model_agb, rrbp_sf, predict_type = "conf_int", std_error = TRUE)

map_ta_ch_num <- process_predictions(final_model_ch, rrta_sf, predict_type = "numeric")
map_ta_ch_ci <- process_predictions(final_model_ch, rrta_sf, predict_type = "conf_int", std_error = TRUE)
map_ta_agb_num <- process_predictions(final_model_agb, rrta_sf, predict_type = "numeric")
map_ta_agb_ci <- process_predictions(final_model_agb, rrta_sf, predict_type = "conf_int", std_error = TRUE)

map_bp_ch_num_resampled <- resample(map_bp_h_num, rr_modelbp, method = "bilinear")                   
map_bp_agb_num_resampled <- resample(map_bp_agb_num, rr_modelbp, method = "bilinear")  
ext_uav <- ext(237660, 239700, 9070750, 9073100)

#Residual: actual values - predicted values
subs_ch = map_bp_ch_num_resampled$.pred-rr_modelbp$Height.q95
subs_ch = rr_modelbp$Height.q95-map_bp_ch_num_resampled$.pred
subs_ch  = subs_ch %>% crop(ext_uav)
subs_agb = map_bp_agb_num_resampled$.pred-rr_modelbp$AGB.Chave 
subs_agb = rr_modelbp$AGB.Chave - map_bp_agb_num_resampled$.pred
subs_agb = subs_agb %>% crop(ext_uav)
terra::plot(subs_ch)

tiff("your/path/Prediction_CH_BP_map.tiff", width = 2000, height = 1600, res = 300,pointsize = 9)  
par(mfrow=c(2,2))
terra::plot(map_bp_h_num, main="Canopy height mean prediction (m)")
terra::plot(boundary, add=T)
terra::plot(map_bp_h_ci[[3]], main="Standard error (m)")
terra::plot(rr_modelbp[[2]], main = 'UAV-LiDAR aggregated H95 (m)')
terra::plot(subs_h, main="Subtraction of LiDAR and prediction value (m)")
dev.off()

tiff("your/path/Prediction_AGB_BP_map.tiff", width = 2600, height = 1800, res = 300,pointsize = 9)  
par(mfrow=c(2,2))
terra::plot(map_bp_agb_num, main="Aboveground biomass mean prediction (Mg/ha)")
terra::plot(boundary, add=T)
terra::plot(map_bp_agb_ci[[3]], main="Standard error (Mg/ha)")
terra::plot(rr_modelbp[[2]], main = 'UAV-LiDAR aggregated AGB (Mg/ha)')
terra::plot(subs_a, main="Subtraction of LiDAR and prediction value (Mg/ha)")
dev.off()

tiff("your/path/Prediction_CH_TA_map.tiff", width = 2000, height = 1600, res = 300,pointsize = 7)  
par(mfrow=c(2,2))
terra::plot(map_ta_h_num, main="Canopy height mean prediction (m)")
terra::plot(map_ta_h_ci[[2]], main="Canopy height P95 prediction (m)")
terra::plot(map_ta_h_ci[[3]], main="Standard error (m)")
dev.off()

tiff("your/path/Prediction_AGB_TA_map.tiff", width = 2000, height = 1600, res = 300,pointsize = 7)  
par(mfrow=c(2,2))
terra::plot(map_ta_agb_num, main="Aboveground biomass mean prediction (Mg/ha)")
terra::plot(map_ta_agb_num, main="Aboveground biomass P95 prediction (Mg/ha)")
terra::plot(map_ta_agb_ci[[3]], main="Standard error (Mg/ha)")
terra::plot(agc_ta, main="Aboveground carbon mean prediction (MgC/ha)")
dev.off()

writeRaster(subs_ch, "your/path/ch_subtract_bp.tif", overwrite = TRUE)
writeRaster(subs_agb, "your/path/agb_subtract_bp.tif", overwrite = TRUE)

writeRaster(map_bp_ch_num, "your/path/ch_num_bp.tif", overwrite = TRUE)
writeRaster(map_bp_ch_ci, "your/path/ch_ci_bp.tif", overwrite = TRUE)

writeRaster(map_ta_ch_num, "your/path/ch_num_ta.tif", overwrite = TRUE)
writeRaster(map_ta_ch_ci, "your/path/ch_ci_ta.tif", overwrite = TRUE)

writeRaster(map_bp_agb_num, "your/path/agb_num_bp.tif", overwrite = TRUE)
writeRaster(map_bp_agb_ci, "your/path/agb_ci_bp.tif", overwrite = TRUE)

writeRaster(map_ta_agb_num, "your/path/agb_num_ta.tif", overwrite = TRUE)
writeRaster(map_ta_agb_ci, "your/path/agb_ci_ta.tif", overwrite = TRUE)


######  RESIDUAL ANALYSIS  ######

# Calculate the residuals between predicted and field CH or AGB in Tahura Ngurah Rai

# Read field data 
fs.data <- st_read("M://Plot_level_CH_AGB.shp")
fs.data <- st_transform(fs.data, crs = 32750)

# Zonal statistics of CH and AGB from predicted raster data
zsdata.ta.ch  <- terra::zonal(map_ta_ch_num,field.ta, fun='mean', as.polygons=TRUE, exact=TRUE, na.rm=TRUE)  %>% 
  sf::st_as_sf(coords = c("x", "y"), crs = 32750)%>%
  mutate(resid_ta_ch = Height.q95 - .pred) %>% terra::vect() %>% terra::buffer(width = 15)
head(zsdata.ta.ch)

zsdata.ta.agb  <- terra::zonal(map_ta_agb_num,field.ta, fun='mean', as.polygons=TRUE, exact=TRUE, na.rm=TRUE)  %>% 
  sf::st_as_sf(coords = c("x", "y"), crs = 32750)%>% 
  mutate(resid_ta_a = AGB.Chave - .pred)%>% terra::vect() %>% terra::buffer(width = 15) 

#writeVector(zsdata.ta.ch, "your/path/TA_plot_predict_CH.shp",overwrite=TRUE)
#writeVector(zsdata.ta.agb, "your/path/TA_plot_predict_AGB.shp",overwrite=TRUE)

zsdata.ta.ch_  <-zsdata.ta.ch %>% as.data.frame()
zsdata.ta.agb_  <-  zsdata.ta.agb%>% as.data.frame()
zsdata.ta.ch.df <- left_join(zsdata.ta,zsdata.ta.ch_, by = "Site")
zsdata.ta.agb.df <- left_join(zsdata.ta,zsdata.ta.agb_, by = "Site")

library(ggpmisc)
library(Rmisc)
library(ggrepel)
conflicted::conflicts_prefer(dplyr::mutate)

# Plot residuals vs. predicted values
resid_plot_ch = ggplot(zsdata.ta.ch.df, aes(x = .pred, y = resid_ta_ch, col=Elevation_Cop30m, shape=Type)) +
  geom_point(show.legend=F,size=2) +ylim(-7,7)+
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  theme(legend.position='none',)+
  paletteer::scale_color_paletteer_c(trans = 'reverse',"grDevices::Spectral")+
  labs(title = "A. Canopy height", x = "Predicted canopy height (m)", y = "Residual", color="Elevation (m)")

resid_plot_agb = ggplot(zsdata.ta.agb.df, aes(x = .pred, y = resid_ta_a, col=Elevation_Cop30m, shape=Type)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_point(size=2) +paletteer::scale_color_paletteer_c(trans = 'reverse',"grDevices::Spectral")+ylim(-480,480)+
  theme(legend.position='none')+#,plot.title = element_text(face = "bold")
  labs(title = 'B. Aboveground biomass', x = expression("Predicted aboveground biomass (Mg ha"^-1 * ")"), 
       y = "Residual",Type="Mangrove type",  color="Elevation (m)")
resid_plot_h+resid_plot_agb 
#ggsave("your/path/TA_H_AGB_residuals.tiff",  width = 9, height = 4, dpi = 400, compression = "lzw")


ch_ta_all = zsdata.ta.ch.df %>%
  mutate(across(c(Type,Age), as.factor)) %>%
  ggplot(aes(x=.pred, y=height_q95)) +
  paletteer::scale_color_paletteer_c(trans = 'reverse',"grDevices::Spectral")+
  geom_point(aes(col=Elevation_Cop30m,shape=Type), size=2.5) + theme_bw()+
  theme_bw()+theme(legend.position='none',plot.title = element_text(size = 14))+
  geom_abline(slope = 1, intercept = 0,linetype = "dashed", color = "red") +  # Perfect fit line
  stat_poly_eq(rr.digits=2, p.digits=2,formula = y ~ x, use_label( "R2", "p", "n"))+
  stat_smooth(method = 'lm',formula =y~x, 
              se = FALSE,col='black')+
  labs(title = "A. Canopy height" , y='Observed (m)', x= "Predicted (m)")+ylim(0,20)+ xlim(0,20)#

ch_ta = zsdata.ta.ch.df %>%
  mutate(across(c(Type,Age), as.factor)) %>%
  mutate(Type = fct_rev(Type))%>%
  ggplot(aes(x=.pred, y=height_q95)) +
  paletteer::scale_color_paletteer_c(trans = 'reverse',"grDevices::Spectral")+
  geom_point(aes(col=Elevation_Cop30m,shape=Type), size=2.5) + 
  theme_bw()+theme(legend.position='none')+
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",color = "red") +  # Perfect fit line
  stat_poly_eq(rr.digits=2, p.digits=2,formula = y ~ x, use_label("R2", "p", "n"))+
  stat_smooth(method = 'lm',formula =y~x, 
              se = FALSE,col='black')+
  labs(title='',  x='Observed (m)', y= "Predicted (m)")+
  facet_wrap(Type~., ncol=2)+ylim(0,20)+ xlim(0,20)

agb_ta_all = zsdata.ta.agb.df %>%
  mutate(across(c(Type,Age), as.factor)) %>%
  ggplot(aes(x=.pred, y=AGB.Chave)) +
  paletteer::scale_color_paletteer_c(trans = 'reverse',"grDevices::Spectral")+
  geom_point(aes(col=Elevation_Cop30m,shape=Type), size=2.5) + theme_bw()+
  theme(legend.position="none",plot.title = element_text(size = 14))+
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",color = "red") +  # Perfect fit line
  stat_poly_eq(rr.digits=2, p.digits=2,formula = y ~ x, use_label( "R2", "p", "n"))+
  stat_smooth(method = 'lm',formula =y~x, 
              se = FALSE,col='black')+
  labs(shape='Mangrove type',color ='Mangrove age (year)',title = 'B. Aboveground biomass',
       y=expression("Observed (Mg ha"^-1 * ")"), 
       x= expression("Predicted (Mg ha"^-1 * ")"))+
  ylim(0,610)+ xlim(0,610)

agb_ta = zsdata.ta.agb.df %>%
  mutate(across(c(Type,Age), as.factor)) %>%
  mutate(Type = fct_rev(Type))%>%
  ggplot(aes(x=.pred, y=AGB.Chave)) +#
  geom_point(aes(col=Elevation_Cop30m,shape=Type), size=2.5) + theme_bw()+
  paletteer::scale_color_paletteer_c(trans = 'reverse',"grDevices::Spectral")+
  theme(legend.position="bottom",legend.box="vertical", legend.margin=margin())+
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",color = "red") +  # Perfect fit line
  stat_poly_eq(rr.digits=2, p.digits=2,formula = y ~ x, use_label( "R2", "p", "n"))+
  stat_smooth(method = 'lm',formula =y~x, 
              se = FALSE,col='black')+
  labs(shape='Mangrove type',color ='Copernicus DEM \n elevation (m)',
       y=expression("Observed (Mg ha"^-1 * ")"), 
       x=expression("Predicted (Mg ha"^-1 * ")"))+
  facet_wrap(Type~., ncol=2)+ylim(0,610)+ xlim(0,610)


h_ta_all+h_ta+agb_ta_all+agb_ta+plot_layout(widths = c(1, 2))
#ggsave("your/path/TA_CH_AGB_predvstrue.tiff",  width = 8.5, height = 7.5, dpi = 400, compression = "lzw")

pg1=resid_plot_h+h_ta_all+h_ta+plot_layout(widths = c(1,1, 2))
pg2=resid_plot_agb+agb_ta_all+agb_ta +plot_layout(widths = c(1,1, 2))
pg1/pg2
#ggsave("your/path/TA_CH_AGB_residual_obs_pred_final__.tiff",  width = 12, height = 7, dpi = 400, compression = "lzw")

############## MAP OF SPATIAL CROSS-VALIDATION ###########

map1=autoplot(rr_block_split1)+ theme_bw()+theme_void()+labs(title='Spatial data partition')+
  scale_fill_manual(values = c('Testing'="lightsalmon", 'Training'="lightblue", 'NA'="skyblue"), #
                    name="Data",labels=c("(1) Model development","(2) Model finalization","Buffer"))+
  scale_color_manual(values = c('Testing'="lightsalmon", 'Training'="lightblue", 'NA'="skyblue"),#
                     name="Data",labels=c("(1) Model development","(2) Model finalization","Buffer"))

map21=autoplot(rr_block_split21)+ theme_bw()+theme_void()+labs(title='(1) Model development')+
  scale_fill_manual(values = c('Testing'="lightsalmon", 'Training'="lightsalmon3",'NA'="skyblue"), #
                    name="Data",labels=c("Feature selection","Hyperparameter tuning", "Buffer"))+
  scale_color_manual(values = c('Testing'="lightsalmon", 'Training'="lightsalmon3",'NA'="skyblue"), #, "Buffer"
                     name="Data",labels=c("Feature selection","Hyperparameter tuning", "Buffer"))

map22=autoplot(rr_block_split22a)+ theme_bw()+theme_void()+labs(title="(2) Model finalization")+
  scale_fill_manual(values = c('Testing'="lightblue", 'Training'="lightblue3", 'NA'="skyblue"), #
                    name="Data",labels=c("Final model testing","Final model training","Buffer"))+
  scale_color_manual(values = c('Testing'="lightblue", 'Training'="lightblue3",'NA'="skyblue"), #, 
                     name="Data",labels=c("Final model testing","Final model training","Buffer"))
map1+map21+map22+plot_layout(ncol=3)
#ggsave("your/path/Spatial partition.tiff",  width = 12, height = 6, dpi = 300)#, compression = "lzw"

sft21=autoplot(spatial_folds_train21)+labs(title='Spatial cross-validation (1)')+theme_bw()+theme_void()+theme(legend.position = "bottom")
sft22=autoplot(spatial_folds_train22_h)+ labs(title='Spatial cross-validation (2)')+theme_bw()+theme_void()+theme(legend.position = "bottom")

sft21+sft22+plot_layout(nrow=1)
#ggsave("your/path/Spatial CV.tiff",  width = 12, height = 8, dpi = 300)#, compression = "lzw"

library(magick)
# Combine images
image1=image_read("your/path/Spatial partition.tiff") 
image2=image_read("your/path/Spatial CV.tiff")

residmapch =image_read("your/path/Residual CH final.tif")
residmapagb =image_read("your/path/Residual AGB final.tif")

# Join the images
rmagick <- c(image1, image2) 
residmagick = c(residmapch, residmapagb)

# Append the images
image_append(rmagick, stack=T)%>%
  image_write("your/path/Spatial partition and CV scheme.tiff", format = "tiff")

image_append(residmagick, stack=T)%>%
  image_write("your/path/Residualmapfinal_STACK.tiff", format = "tiff")

################ UAV-LIDAR POINT CLOUDS OF MANGROVE RESTORATION ####################

library(rGEDI)
library(rGEDIsimulator)
library(lidR)
library(plot3D)

clipA = readLAS("your/path/A.Early-stage non-restoration.las")
clipB = readLAS("your/path/B.Early-stage restoration.las")
clipC = readLAS("your/path/C.Nypa mangroves.las")
clipD = readLAS("your/path/D.Late-stage restoration.las")
clipE = readLAS("your/path/E.Late-stage non-restoration.las")

# Example of UAV-LiDAR plot
plot(clipA)

# Graph of UAV-LiDAR point clouds
tiff("your/path/uav_lidar_simulation.tiff", width = 3500, height = 800, units = 'px', res = 300, compression = "lzw") 

par(mfrow=c(1,5), mar=c(0.1,1,2,0), oma=c(0,2,0,0),cex.axis = 1)
scatter3D(clipA@data$X,clipA@data$Y,clipA@data$Z,pch = 16,colkey = FALSE, main="(A)\n Early-stage non-restoration", 
          bty = "u",col.panel ="gray95",phi = 30,alpha=1,theta=45, zlim = c(-2, 18),cex.lab=0.9,
          col.grid = "gray50", xlab="UTM Easting (m)", ylab="UTM Northing (m)", zlab="Normalized height (m)")
scatter3D(clipB@data$X,clipB@data$Y,clipB@data$Z,pch = 16,colkey = FALSE, main="(B)\n Early-stage restoration",
          bty = "u",col.panel ="gray95",phi = 30,alpha=1,theta=45,zlim = c(-4, 18),cex.lab=0.9,
          col.grid = "gray50", xlab="UTM Easting (m)", ylab="UTM Northing (m)", zlab="Normalized height (m)")
scatter3D(clipC@data$X,clipC@data$Y,clipC@data$Z,pch = 16,colkey = FALSE, main="(C)\n Nypa mangroves",
          bty = "u",col.panel ="gray95",phi = 30,alpha=1,theta=45,zlim = c(-4, 18),cex.lab=0.9,
          col.grid = "gray50", xlab="UTM Easting (m)", ylab="UTM Northing (m)", zlab="Normalized height (m)")
scatter3D(clipD@data$X,clipD@data$Y,clipD@data$Z,pch = 16,colkey = FALSE, main="(D)\n Late-stage restoration",
          bty = "u",col.panel ="gray95",phi = 30,alpha=1,theta=45,zlim = c(-4, 18),cex.lab=0.9,
          col.grid = "gray50", xlab="UTM Easting (m)", ylab="UTM Northing (m)", zlab="Normalized height(m)")
scatter3D(clipE@data$X,clipE@data$Y,clipE@data$Z,pch = 16,colkey = FALSE, main="(E)\n Late-stage non-restoration",
          bty = "u",col.panel ="gray95",phi = 30,alpha=1,theta=45,zlim = c(-4, 18),cex.lab=0.9,
          col.grid = "gray50", xlab="UTM Easting (m)", ylab="UTM Northing (m)", zlab="Normalized height (m)")

dev.off()



