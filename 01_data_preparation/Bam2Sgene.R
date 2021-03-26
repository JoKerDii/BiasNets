## from bam to single isoform gene

# file = "SRR5968905_s"
# dir = "bwa_scRNA"
# bam.file <- paste("/home/zhendi/wei/",dir,'/',file,".bam",sep = "")

Bam2Sgene <- function (bam.file){
  #from bam to single isoform gene
  
  txdb <- TxDb.Hsapiens.UCSC.hg19.knownGene
  map_tx <- select(TxDb.Hsapiens.UCSC.hg19.knownGene,
                   keys(TxDb.Hsapiens.UCSC.hg19.knownGene,"GENEID"), 
                   keytype = "GENEID",
                   columns = "TXID")
  single_isoform_genes <- setdiff(map_tx$GENEID, map_tx$GENEID[duplicated(map_tx$GENEID)])
  single_isoform_tx <- map_tx$TXID[na.omit(match(map_tx$GENEID, single_isoform_genes))]
  genes <- exonsBy(txdb,by="tx")[single_isoform_tx]
  genes <- keepStandardChromosomes(genes, pruning.mode="coarse")
  genes
}

# genes <- Bam2Sgene(bam.file)
# genes