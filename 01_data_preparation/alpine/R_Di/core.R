# genes <- genes[sum(width(genes)) >= readlength]
# exons = genes[[gene.names[[1]]]]

buildFragtypes <- function(exons, genome, readlength,
                           # minsize, maxsize, 
                           gc=TRUE, gc.str=TRUE, vlmm=TRUE) {
  stopifnot(is(exons,"GRanges"))
  stopifnot(is(genome,"BSgenome"))
  # stopifnot(is.numeric(minsize) & is.numeric(maxsize) & is.numeric(readlength))
  # stopifnot(sum(width(exons)) >= maxsize)
  stopifnot(all(c("exon_rank","exon_id") %in% names(mcols(exons))))
  stopifnot(!any(strand(exons) == "*"))
  
  # these parameters must be fixed, as dictated by fitVLMM()
  npre <- 8
  npost <- 12
  
  map <- mapTxToGenome(exons) # sometimes smaller than median fragment length
  l <- nrow(map) # should be the entire transcript
  strand <- as.character(strand(exons)[1])
  start <- rep(seq_len(l-readlength + 1), each=1) # Q: fragment can exceed the tx.
  end <- as.integer(start + readlength - 1) #
  # start <- rep(seq_len(l-minsize+1),each=maxsize-minsize+1)
  # end <- as.integer(start + minsize:maxsize - 1)
  mid <- as.integer(0.5 * (start + end))
  relpos <- mid/l
  fraglen <- as.integer(end - start + 1)
  id <- IRanges(start, end)
  fragtypes <- DataFrame(start=start,end=end,relpos=relpos,fraglen=fraglen,id=id)
  fragtypes <- fragtypes[fragtypes$end <= l,,drop=FALSE]
  exon.dna <- getSeq(genome, exons)
  tx.dna <- unlist(exon.dna)
  
  # get the GC content for the entire fragment
  gc.vecs <- lapply(readlength, function(i) {
    DNAStringSet( Views(tx.dna, slidingWindows(IRanges(start = 1, end = length(tx.dna)), i)[[1]]))
  })
  fragtypes <- fragtypes[order(fragtypes$fraglen),,drop=FALSE]
  fragtypes$gc <- do.call(c, gc.vecs)
  fragtypes <- fragtypes[order(fragtypes$start),,drop=FALSE]
  
  
  # if (vlmm) {
  #   # strings needed for VLMM
  #   fragtypes$fivep.test <- fragtypes$start - npre >= 1
  #   fragtypes$fivep <- as(Views(tx.dna, fragtypes$start - ifelse(fragtypes$fivep.test, npre, 0),
  #                               fragtypes$start + npost), "DNAStringSet")
  #   fragtypes$threep.test <- fragtypes$end + npre <= length(tx.dna)
  #   fragtypes$threep <- as(Views(tx.dna, fragtypes$end - npost,
  #                                fragtypes$end + ifelse(fragtypes$threep.test, npre, 0),),
  #                          "DNAStringSet")
  #   # reverse complement the three prime sequence
  #   fragtypes$threep <- reverseComplement(fragtypes$threep)
  # }
  
  
  # if (gc.str) {
  #   # additional features: GC in smaller sections
  #   gc.40 <- as.numeric(letterFrequencyInSlidingView(tx.dna, 40, letters="CG", as.prob=TRUE))
  #   max.gc.40 <- max(Views(gc.40, fragtypes$start, fragtypes$end - 40 + 1))
  #   gc.20 <- as.numeric(letterFrequencyInSlidingView(tx.dna, 20, letters="CG", as.prob=TRUE))
  #   max.gc.20 <- max(Views(gc.20, fragtypes$start, fragtypes$end - 20 + 1))
  #   fragtypes$GC40.90 <- as.numeric(max.gc.40 >= 36/40)
  #   fragtypes$GC40.80 <- as.numeric(max.gc.40 >= 32/40)
  #   fragtypes$GC20.90 <- as.numeric(max.gc.20 >= 18/20)
  #   fragtypes$GC20.80 <- as.numeric(max.gc.20 >= 16/20)
  # }
  # these are the fragment start and end in genomic space
  # so for minus strand tx, gstart > gend
  # fragtypes$gstart <- txToGenome(fragtypes$start, map)
  # fragtypes$gend <- txToGenome(fragtypes$end, map)
  # fragtypes$gread1end <- txToGenome(fragtypes$start + readlength - 1, map)
  # fragtypes$gread2start <- txToGenome(fragtypes$end - readlength + 1, map)
  #message("nrow fragtypes: ",nrow(fragtypes))
  fragtypes
}

######### unexported core functions #########

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
             exon_rank=rep(exons$exon_rank, width(exons)))
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
    reads[[i]] <- IRanges(start[valid], end[valid])
  }
  names(reads) <- names(grl)
  reads
}


matchReadsToFraglist <- function(reads, fraglist) { # count reads here
  for (tx.idx in seq_along(fraglist)) {
    uniq.reads <- unique(reads[[tx.idx]])
    readtab <- table(match(reads[[tx.idx]], uniq.reads))
    fraglist[[tx.idx]]$count <- 0
    # this can be slow (up to 1 min) when fraglist has many millions of rows
    # match.uniq <- match(uniq.reads, fraglist[[tx.idx]]$id)
    match.uniq <- match(start(uniq.reads), start(fraglist[[tx.idx]]$id))
    reads.in.fraglist <- !is.na(match.uniq)
    # uniq.reads <- uniq.reads[reads.in.fraglist] # not needed
    readtab <- readtab[reads.in.fraglist]
    # the map between {uniq.reads that are in fraglist} and {rows of fraglist}
    match.uniq.non.na <- match.uniq[!is.na(match.uniq)]
    fraglist[[tx.idx]][match.uniq.non.na,"count"] <- as.numeric(readtab)
    fraglist[[tx.idx]][,"count_overlap"] <- countOverlaps(fraglist[[tx.idx]]$id, reads[[tx.idx]])
  }
  fraglist
}

# Zhen Wei
countReadsToFraglist <- function(reads, fraglist) { # count reads here
  for (tx.idx in seq_along(fraglist)) {
    fraglist[[tx.idx]][,"count_overlap"] <- countOverlaps(fraglist[[tx.idx]]$id, reads[[tx.idx]])
  }
  fraglist
}


subsetAndWeightFraglist <- function(fraglist, downsample=20, minzero=2000) {
  unique.zero.list <- list()
  for (tx in seq_len(length(fraglist))) {
    # need to make a unique id for each fragment
    fraglist[[tx]]$genomic.id <- str_c(fraglist[[tx]]$gstart,"-",
                                       fraglist[[tx]]$gread1end,"-",
                                       fraglist[[tx]]$gread2start,"-",
                                       fraglist[[tx]]$gend)
    unique.zero.list[[tx]] <- fraglist[[tx]]$genomic.id[fraglist[[tx]]$count == 0]
  }
  unique.zero <- unique(do.call(c, unique.zero.list))
  sumzero <- length(unique.zero)
  numzero <- round(sumzero / downsample)
  numzero <- max(numzero, minzero)
  numzero <- min(numzero, sumzero)
  unique.ids <- sample(unique.zero, numzero, replace=FALSE)
  # once again, this time grab all fragments with positive count or in our list of zeros
  fraglist.sub <- list()
  for (tx in seq_len(length(fraglist))) {
    idx.pos <- which(fraglist[[tx]]$count > 0)
    idx.zero <- which(fraglist[[tx]]$genomic.id %in% unique.ids)
    fraglist.sub[[tx]] <- fraglist[[tx]][c(idx.pos,idx.zero),,drop=FALSE]
  }
  fragtypes <- do.call(rbind, fraglist.sub)
  # the zero weight is the number of unique zero count fragtypes in the original fraglist
  # divided by the current (down-sampled) number of zero count fragtypes
  zero.wt <- sumzero / numzero
  # return fragtypes, but with duplicate rows for selected fragments
  fragtypes$wts <- rep(1, nrow(fragtypes))
  fragtypes$wts[fragtypes$count == 0] <- zero.wt
  fragtypes
}

matchToDensity <- function(x, d) {
  idx <- cut(x, c(-Inf, d$x, Inf))
  pdf <- c(0, d$y)
  pdf.x <- pdf[ idx ] + 1e-6
  stopifnot(all(pdf.x > 0))
  pdf.x
}
getFPBP <- function(genes, bam.file) {
  gene.ranges <- unlist(range(genes))
  gene.lengths <- sum(width(genes))
  res <- countBam(bam.file, param=ScanBamParam(which=gene.ranges))
  # two records per fragment
  out <- (res$records / 2)/gene.lengths
  names(out) <- names(genes)
  out
}
getLogLambda <- function(fragtypes, models, modeltype, fitpar, bamname) {

  # knots and boundary knots need to come from the fitted parameters object
  # (just use the first sample, knots will be the same across samples)
  model.params <- fitpar[[1]][["model.params"]]
  stopifnot(!is.null(model.params))
  
  gc.knots <- model.params$gc.knots
  gc.bk <- model.params$gc.bk
  relpos.knots <- model.params$relpos.knots
  relpos.bk <- model.params$relpos.bk

  # which formula to use
  f <- models[[modeltype]]$formula

  offset <- numeric(nrow(fragtypes))
  if ("fraglen" %in% models[[modeltype]]$offset) {
    # message("-- fragment length correction")
    offset <- offset + fragtypes$logdfraglen
  }
  if ("vlmm" %in% models[[modeltype]]$offset) {
    # message("-- VLMM fragment start/end correction")
    offset <- offset + fragtypes$fivep.bias + fragtypes$threep.bias
  }
  if (!is.null(f)) {
    stopifnot(modeltype %in% names(fitpar[[bamname]][["coefs"]]))
    # assume: no intercept in formula
    # sparse.model.matrix produces different column names, so don't use
    # mm.big <- sparse.model.matrix(f, data=fragtypes)
    mm.big <- model.matrix(formula(f), data=fragtypes)
    beta <- fitpar[[bamname]][["coefs"]][[modeltype]]
    stopifnot(any(colnames(mm.big) %in% names(beta)))
    if (all(is.na(beta))) stop("all coefs are NA")
    beta[is.na(beta)] <- 0 # replace NA coefs with 0: these were not observed in the training data
    # this gets rid of the gene1, gene2 and Intercept terms
    beta <- beta[match(colnames(mm.big), names(beta))]
    # add offset
    log.lambda <- as.numeric(mm.big %*% beta) + offset
  } else {
    log.lambda <- offset
  }
  if (!all(is.finite(log.lambda))) stop("log.lambda is not finite")
  log.lambda
}
namesToModels <- function(model.names, fitpar) {
  # create the model.bank
  model.bank <- c(fitpar[[1]][["models"]],
                  list("null"=list(formula=NULL, offset=NULL),
                       "fraglen"=list(formula=NULL, offset="fraglen"),
                       "vlmm"=list(formula=NULL, offset="vlmm"),
                       "fraglen.vlmm"=list(formula=NULL, offset=c("fraglen","vlmm"))))
  models <- model.bank[model.names]
  # replace '+ gene' with '+ 0' in formula
  for (m in model.names) {
    if (!is.null(models[[m]]$formula)) {
      if (!grepl("\\+ gene$",models[[m]]$formula)) {
        stop("was expecting '+ gene' to be at the end of the formula string from fitpar")
      }
      models[[m]]$formula <- sub("\\+ gene$","\\+ 0",models[[m]]$formula)
    }
  }
  models
}

