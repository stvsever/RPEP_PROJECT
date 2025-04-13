import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.pandas2ri as pandas2ri

# Activate automatic conversion between pandas and R dataframes
pandas2ri.activate()

ro.r('''
# Load required libraries
library(tidyverse)
library(psych)
library(writexl)  # For Excel export

cat("STAP 1: Inlezen van de data...\\n")
data_long <- read.csv("/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/concatenated_data.csv",
                      stringsAsFactors = FALSE)

cat("Kolommen in de dataset:\\n")
print(names(data_long))
cat("\\n")

cat("STAP 2: Long naar wide transformatie...\\n")
data_wide <- data_long %>%
  pivot_wider(
    names_from = condition_item,
    values_from = affinity_score,
    id_cols = participant_identification,
    values_fn = mean
  )

cat("Voorbeeld van de getransformeerde data:\\n")
print(head(data_wide))
cat("\\n")

cat("STAP 3: Itemsets detecteren...\\n")
all_items <- setdiff(names(data_wide), "participant_identification")
F_items <- grep("^F[0-9]+", all_items, value = TRUE)
DFU_items <- grep("^DFU[0-9]+", all_items, value = TRUE)

cat("F-items:\\n")
print(F_items)
cat("DFU-items:\\n")
print(DFU_items)
cat("\\n")

cat("STAP 4: Gemiddelde scores berekenen per participant...\\n")
data_wide <- data_wide %>%
  mutate(
    mean_all = rowMeans(across(all_of(all_items)), na.rm = TRUE),
    mean_F = rowMeans(across(all_of(F_items)), na.rm = TRUE),
    mean_DFU = rowMeans(across(all_of(DFU_items)), na.rm = TRUE)
  )
print(head(data_wide))
cat("\\n")

cat("STAP 5: Factoranalyse uitvoeren (1 factor per set)...\\n\\n")

cat("--- FA: Alle items ---\\n")
fa_all <- fa(data_wide[all_items], nfactors = 1, rotate = "none", scores = "regression")
print(fa_all)
cat("\\n")

cat("--- FA: Alleen F-items ---\\n")
fa_F <- fa(data_wide[F_items], nfactors = 1, rotate = "none", scores = "regression")
print(fa_F)
cat("\\n")

cat("--- FA: Alleen DFU-items ---\\n")
fa_DFU <- fa(data_wide[DFU_items], nfactors = 1, rotate = "none", scores = "regression")
print(fa_DFU)
cat("\\n")

cat("STAP 6: Factor scores toevoegen aan dataset...\\n")
data_wide$fa_all <- fa_all$scores
data_wide$fa_F <- fa_F$scores
data_wide$fa_DFU <- fa_DFU$scores
print(head(data_wide))
cat("\\n")

cat("STAP 7: Correlaties tussen gemiddelde en factor scores...\\n")
cor_all <- cor(data_wide$mean_all, data_wide$fa_all, use = "pairwise.complete.obs")
cor_F <- cor(data_wide$mean_F, data_wide$fa_F, use = "pairwise.complete.obs")
cor_DFU <- cor(data_wide$mean_DFU, data_wide$fa_DFU, use = "pairwise.complete.obs")

cat(sprintf("Correlatie mean_all ~ fa_all: %.3f\\n", cor_all))
cat(sprintf("Correlatie mean_F ~ fa_F: %.3f\\n", cor_F))
cat(sprintf("Correlatie mean_DFU ~ fa_DFU: %.3f\\n", cor_DFU))
cat("\\n")

cat("STAP 8: Export naar Excel...\\n")
output_data <- data_wide %>%
  select(participant_identification, mean_all, mean_F, mean_DFU, fa_all, fa_F, fa_DFU)

write_xlsx(
  output_data,
  path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment/experiment/data/preprocessed/latent_disposition.xlsx"
)

cat("✅ Bestand opgeslagen als 'latent_disposition.xlsx' in de preprocessed-directory.\\n")
''')
