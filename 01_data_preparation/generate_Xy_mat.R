
file = "SRR5968898_s"
dir = "bwa_scRNA"
readlength <- 216

library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(BSgenome.Hsapiens.UCSC.hg19)
library(Homo.sapiens)
library(GenomicAlignments)
library(GenomicRanges)
library(rtracklayer)
library(stringr)

source('~/wei/scripts/Bam2Sgene.R')
source('~/wei/scripts/alpine/R_Di/core.R') # /R_Di/core.R <= modified
source('~/wei/scripts/alpine/R/estimate_abundance.R')
source('~/wei/scripts/alpine/R_Di/fit_bias.R') # /R_Di/fit_bias.R <= modified
source('~/wei/scripts/alpine/R/helper.R')
source('~/wei/scripts/alpine/R/plots.R')
source('~/wei/scripts/alpine/R/predict.R')
source('~/wei/scripts/alpine/R/vlmm.R')

bam.file <- paste("/home/zhendi/wei/",dir,'/',file,".bam",sep = "")
genes <- Bam2Sgene(bam.file)
genes <- genes[sum(width(genes)) >= readlength]
gene.names <- names(genes)
genome <- BSgenome.Hsapiens.UCSC.hg19

names(gene.names) <- gene.names
fragtypes <- lapply(gene.names, function(gene.name) {
  buildFragtypes(genes[[gene.name]],
                 genome, readlength)
})
# fragtypes 

char <- data.frame()
for (i in seq_along(fragtypes)) {
  character <- data.frame(as.character(fragtypes[[i]]$gc))
  char <- rbind(char,character)
}

Feature_M <- matrix(unlist(stringr::str_split(char[[1]],"")), byrow = TRUE, nrow = length(char[[1]]))
Feature_M[Feature_M=="A"] <- 0
Feature_M[Feature_M=="T"] <- 1
Feature_M[Feature_M=="C"] <- 2
Feature_M[Feature_M=="G"] <- 3
Feature_M <- apply(Feature_M,2,as.numeric)

library(splines)
library(speedglm)
fitpar <- fitBiasModels(genes=genes[gene.names],
                        bam.file=bam.file,
                        fragtypes=fragtypes,
                        genome=genome,
                        readlength=readlength)
# fitpar
counts <- data.frame()
counts_overlap <- data.frame()

for (i in seq_along(fitpar)) {
  count <- data.frame(fitpar[[i]]$count)
  counts <- rbind(counts,count)
  
  count_overlap <- data.frame(fitpar[[i]]$count_overlap)
  counts_overlap <- rbind(counts_overlap,count_overlap)
}

Feature_M <- cbind(Feature_M, counts = counts)
Feature_M <- cbind(Feature_M, counts_overlap = counts_overlap)

print(paste(" Generate Sequence for: ", dir,'/',file,".bam"," ", sep = ""))
# saveRDS(char, paste("/home/zhendi/wei/",dir,'/seq/',file, ".rds", sep = ""))
# saveRDS(counts, paste("/home/zhendi/wei/",dir,'/seq/',file, "_y.rds", sep = ""))
# saveRDS(counts, paste("/home/zhendi/wei/",dir,'/seq/',file, "_y.rds", sep = ""))
# write.csv(Feature_M, paste("/data/zhendi/wei/",dir,'/seq/',file, ".csv", sep = ""), row.names = FALSE)
saveRDS(Feature_M, paste("/data/zhendi/wei/",dir,'/seq/',file, ".rds", sep = ""))