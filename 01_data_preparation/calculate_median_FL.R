file = "SRR5968913_s"
dir = "bwa_scRNA"

##=====================================================================
## Zhen's Comments:
##=====================================================================
## Use readGAlignAlpine() to read in the bam files.
## make sure to refer to alpine's source code, not refer to the homework of BIO214
alpineFlag <- function() scanBamFlag(isSecondaryAlignment=FALSE)
readGAlignAlpine <- function(bam.file, generange, manual=TRUE) {
  if (manual) {
    param <- ScanBamParam(which=generange, what=c("flag","mrnm","mpos"), flag=alpineFlag())
    gal <- readGAlignments(bam.file, use.names=TRUE, param=param)
    makeGAlignmentPairs(gal)
  } else {
    readGAlignmentPairs(bam.file,param=ScanBamParam(which=generange,flag=alpineFlag()))
  }
}
##=====================================================================
## Comments end
##=====================================================================
startLeft <- function(x) {
  first.plus <- as.logical(strand(first(x)) == "+")
  ifelse(first.plus, start(first(x)), start(last(x)))
}
endRight <- function(x) {
  first.plus <- as.logical(strand(first(x)) == "+")
  ifelse(first.plus, end(last(x)), end(first(x)))
}
mapTxToGenome <- function(exons) {
  strand <- as.character(strand(exons)[1])
  stopifnot(all(exons$exon_rank == seq_along(exons)))
  bases <- S4Vectors:::fancy_mseq(width(exons), start(exons)-1L,
                                  rev=(strand == "-"))
  data.frame(tx=seq_along(bases),
             genome=bases,
             exon_rank=rep(exons$exon_rank, width(exons))) # error
}
genomeToTx <- function(genome, map) map$tx[match(genome, map$genome)]
txToGenome <- function(tx, map) map$genome[match(tx, map$tx)]
txToExon <- function(tx, map) map$exon_rank[match(tx, map$tx)]

gaToReadsOnTx <- function(ga, grl, fco=NULL) {
  reads <- list()
  for (i in seq_along(grl)) {
    exons <- grl[[i]]
    strand <- as.character(strand(exons)[1])
    read.idx <- if (is.null(fco)) {
      seq_along(ga)
    } else {
      queryHits(fco)[subjectHits(fco) == i]
    }
    map <- mapTxToGenome(exons)
    # depending on strand of gene:
    # start of left will be the first coordinate on the transcript (+ gene)
    # or start of left will be the last coordinate on the transcript (- gene)
    if (strand == "+") {
      start <- genomeToTx(startLeft(ga[read.idx]), map)
      end <- genomeToTx(endRight(ga[read.idx]), map)
    } else if (strand == "-") {
      start <- genomeToTx(endRight(ga[read.idx]), map)
      end <- genomeToTx(startLeft(ga[read.idx]), map)
    }
    valid <- start < end & !is.na(start) & !is.na(end)
    reads[[i]] <- IRanges(start[valid], end[valid]) # error
  }
  names(reads) <- names(grl)
  reads
}

######################################################################

library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(Homo.sapiens)
library(GenomicAlignments)
library(GenomicRanges)

# bam.file = "/Users/zhen/Dropbox/BiasNet_local/SRR5968905_s.bam"
# bam.file = "/home/zhendi/wei/star_RNAseq/SRR5968905_s.bam"
# file = "SRR5968905_s"
# dir = "bwa_scRNA"
bam.file <- paste("/home/zhendi/wei/",dir,'/',file,".bam",sep = "")

txdb <- TxDb.Hsapiens.UCSC.hg19.knownGene

map_tx <- select(TxDb.Hsapiens.UCSC.hg19.knownGene,
                 keys(TxDb.Hsapiens.UCSC.hg19.knownGene,"GENEID"), 
                 keytype = "GENEID",
                 columns = "TXID")
single_isoform_genes <- setdiff(map_tx$GENEID, map_tx$GENEID[duplicated(map_tx$GENEID)])
single_isoform_tx <- map_tx$TXID[na.omit(match(map_tx$GENEID, single_isoform_genes))]
genes <- exonsBy(txdb,by="tx")[single_isoform_tx]
genes <- keepStandardChromosomes(genes, pruning.mode="coarse")


reads <- list()
for (i in seq_along(genes)){
  gene.name <- names(genes)[i]
  gene <- genes[[gene.name]]
  generange <- range(gene)
  generange <- keepStandardChromosomes(generange)
  # strand(generange) <- '*' # ?
  suppressWarnings({
    ga <-readGAlignAlpine(bam.file, generange)
  })
  fco <- findCompatibleOverlaps(ga, GRangesList(gene))
  reads[gene.name] <- gaToReadsOnTx(ga, GRangesList(gene), fco) 
}


median_frag_len <- median(width(unlist(IRangesList(reads))))
print(paste(dir,'/',file,".bam"," ",median_frag_len, sep = ""))
saveRDS(median_frag_len, paste("/home/zhendi/wei/",dir,'/',file, ".rds", sep = ""))