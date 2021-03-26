library(dplyr)
gffff <- read.table("GRCh37_RefSeq_24.gff",sep="\t",header=FALSE)
gffff <- data.frame(gffff)
gfffd <- gffff %>% filter(V1 %in% c('chr1','chr2','chr3','chr4','chr5',
                                    'chr6','chr7','chr8','chr9','chr10',
                                    'chr11','chr12','chr13','chr14','chr15',
                                    'chr16','chr17','chr18','chr19','chr20',
                                    'chr21','chr22','chrX','chrY','chrM'))
write.table(gfffd,"GRCh37_RefSeq_24_cleaned.gff",sep="\t",row.names=FALSE,col.names=FALSE,quote = FALSE)
