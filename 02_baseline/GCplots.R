library(ggplot2)
library(dplyr)
# library(ggsci)

# parameters
seq = "star_RNAseq"
data = "ERR188345Aligned.sortedByCoord.out"
root = "/data/zhendi/wei"
readlength <- 163

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


# plot and save
g1 <- ggplot(df, aes(x = GC)) + 
  geom_density() + 
  theme(text = element_text(size=14)) +
  labs(x = "Fragement GC Content", y = "Density")
g1path = paste(root, "/baseline/", seq, "/", data, "/GC_density.png", sep = "")
ggsave(filename = g1path, g1, width = 10, height = 8, dpi = 150, units = "in", device='png')

poisson_smooth <- function(...){
  geom_smooth(method = 'glm', method.args = list(family = 'poisson'), ...)
}

g2 <- ggplot(df, aes(x = GC, y = count_5)) + 
  poisson_smooth(formula = y ~ splines::ns(x, 5)) + 
  theme(text = element_text(size=14)) +
  labs(x = "Fragment GC content", 
       y = "Fragment 5 prime count",
       main = "Poisson Regression")
g2path = paste(root, "/baseline/", seq, "/", data, "/poisson_curve_count_5.png", sep = "")
ggsave(filename = g2path, g2, width = 10, height = 8, dpi = 150, units = "in", device='png')

g3 <- ggplot(df, aes(x = GC, y = count_overlap)) + 
  poisson_smooth(formula = y ~ splines::ns(x, 5)) + 
  theme(text = element_text(size=14)) +
  labs(x = "Fragment GC content", 
       y = "Fragment overlap count",
       main = "Poisson Regression")
g3path = paste(root, "/baseline/", seq, "/", data, "/poisson_curve_count_overlap.png", sep = "")
ggsave(filename = g3path, g3, width = 10, height = 8, dpi = 150, units = "in", device='png')


print('completed')

