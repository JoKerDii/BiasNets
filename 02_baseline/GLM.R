library(ggplot2)
library(dplyr)
# library(ggsci)

# parameters
# seq = "star_RNAseq"
seq = "bwa_scRNA"
# data = "ERR188345Aligned.sortedByCoord.out"
data = "SRR5968867_s"
root = "/data/zhendi/wei"
readlength <- 123

# load data
data_path <- paste(root, '/', seq, "/seq/",  data, ".rds", sep = "") # 162
# data_path <- "/data/zhendi/wei/star_RNAseq/seq/ERR188345Aligned.sortedByCoord.out.rds" # 163
# data_path <- "/data/zhendi/wei/bwa_scRNA/seq/SRR5959996_s.rds" # 214
# data_path <- "/data/zhendi/wei/bwa_scRNA/seq/SRR5968867_s.rds" # 123
df <- readRDS(data_path)

# down sampling
set.seed(seed = 1)
idx <- sample(x = 1:nrow(df),size = nrow(df)*0.01, replace = FALSE)
df <- df[idx, ]

# rename labels
df <- df %>% rename(count_5 = fitpar..i...count, count_overlap = fitpar..i...count_overlap)

# calculating GC content
features <- df[,1:readlength]
features_new <- apply(features, 2, function(x) ifelse(x > 1, 1, 0))
df$GC <- rowSums(features_new)/readlength

# model
glm.GC <- glm(df$count_overlap ~ splines::ns(df$GC,df=5), family = poisson())
summary(glm.GC)
