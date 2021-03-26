
fitBiasModels <- function(genes, bam.file, fragtypes, genome,
                          # models, 
                          readlength, 
                          # minsize, maxsize,
                          speedglm=TRUE,
                          gc.knots=seq(from=.4, to=.6, length=3),
                          gc.bk=c(0,1),
                          relpos.knots=seq(from=.25, to=.75, length=3),
                          relpos.bk=c(0,1)) {
  
  stopifnot(file.exists(bam.file))
  stopifnot(file.exists(paste0(as.character(bam.file),".bai")))
  stopifnot(is(genes, "GRangesList"))
  # stopifnot(all(!is.na(sapply(models, function(x) x$formula))))
  stopifnot(is.numeric(readlength) & length(readlength) == 1)
  stopifnot(all(names(genes) %in% names(fragtypes)))
  
  # if (any(sapply(models, function(m) "vlmm" %in% m$offset))) {
  #   stopifnot("fivep" %in% colnames(fragtypes[[1]]))
  # }
  # for (m in models) {
  #   if (!is.null(m$formula)) {
  #     stopifnot(is.character(m$formula))
  #     if (!grepl("+ gene$",m$formula)) {
  #       stop("'+ gene' needs to be at the end of the formula string")
  #     }
  #   }
  # }
  exon.dna <- getSeq(genome, genes)
  gene.seqs <- as(lapply(exon.dna, unlist), "DNAStringSet")
  # FPBP needed to downsample to a target fragment per kilobase
  # fpbp <- getFPBP(genes, bam.file)

  # TODO check these downsampling parameters now that subset 
  # routine is not related to number of positive counts
  
  # want ~1000 rows per gene, so ~300 reads per gene
  # so ~300/1500 = 0.2 fragments per basepair 
  # target.fpbp <- 0.4
  fitpar.sub <- list()
  fitpar.sub[["coefs"]] <- list()
  fitpar.sub[["summary"]] <- list()
  # create a list over genes, populated with read info from this 'bam.file'
  # so we create a new object, and preserve the original 'fragtypes' object
  fragtypes.sub.list <- list()
  for (i in seq_along(genes)) { 
    
    gene.name <- names(genes)[i]
    gene <- genes[[gene.name]]
    l <- sum(width(gene))
    # add counts per sample and subset
    generange <- range(gene)
    # strand(generange) <- "*" # not necessary # ?????????????
    if (!as.character(seqnames(generange)) %in% seqlevels(BamFile(bam.file))) next
    # this necessary to avoid hanging on highly duplicated regions
    ## roughNumFrags <- countBam(bam.file, param=ScanBamParam(which=generange))$records/2
    ## if (roughNumFrags > 10000) next
    suppressWarnings({
                       ga <- readGAlignAlpine(bam.file, generange)
                     })
    # if (length(ga) < 20) next
    ga <- keepSeqlevels(ga, as.character(seqnames(gene)[1]))
    # downsample to a target FPBP
    # nfrags <- length(ga)
    # this.fpbp <- nfrags / l
    # if (this.fpbp > target.fpbp) {
    #   ga <- ga[sample(nfrags, round(nfrags * target.fpbp / this.fpbp), FALSE)]
    # } 
    ga[strand(ga)=="*"] <- NULL
    fco <- findCompatibleOverlaps(ga, GRangesList(gene)) ## error: some alignment in 'query' have ranges on both strands
    # message("-- ",round(length(fco)/length(ga),2)," compatible overlaps")
    # as.numeric(table(as.character(strand(ga))[queryHits(fco)])) # strand balance
    reads <- gaToReadsOnTx(ga, GRangesList(gene), fco) 
    # fraglist.temp is a list of length 1
    # ...(matchReadsToFraglist also works for multiple transcripts)
    # it will only last for a few lines...
    fraglist.temp <- matchReadsToFraglist(reads, fragtypes[gene.name]) 
    # # remove first and last bp for fitting the bias terms
    # not.first.or.last.bp <- !(fraglist.temp[[1]]$start == 1 | fraglist.temp[[1]]$end == l)
    # fraglist.temp[[1]] <- fraglist.temp[[1]][not.first.or.last.bp,]
    # if (sum(fraglist.temp[[1]]$count) < 20) next
    # randomly downsample and up-weight
    # fragtypes.sub.list[[gene.name]] <- subsetAndWeightFraglist(fraglist.temp,
    #                                                            downsample=200,
    #                                                            minzero=700)
    fragtypes.sub.list[gene.name] <- fraglist.temp
    
  }
  
  
  
  # if (length(fragtypes.sub.list) == 0) stop("not enough reads to model: ",bam.file)
  # # collapse the list over genes into a
  # # single DataFrame with the subsetted and weighted
  # # potential fragment types from all genes
  # # message("num genes w/ suf. reads: ",length(fragtypes.sub.list))
  # if (length(fragtypes.sub.list) < 2) stop("requires at least two genes to fit model")
  # gene.nrows <- sapply(fragtypes.sub.list, nrow)
  # # message("mean rows per gene: ", round(mean(gene.nrows)))
  # # a DataFrame of the subsetted fragtypes
  # fragtypes.sub <- do.call(rbind, fragtypes.sub.list)
  # 
  # # check the FPBP after downsampling:
  # ## gene.counts <- sapply(fragtypes.sub.list, function(x) sum(x$count))
  # ## gene.lengths <- sum(width(genes))
  # ## round(unname(gene.counts / gene.lengths[names(gene.counts)]), 2)
  # 
  # # save the models and parameters
  # fitpar.sub[["models"]] <- models
  # fitpar.sub[["model.params"]] <- list(
  #   readlength=readlength,
  #   # minsize=minsize,
  #   # maxsize=maxsize,
  #   gc.knots=gc.knots,
  #   gc.bk=gc.bk,
  #   relpos.knots=relpos.knots,
  #   relpos.bk=relpos.bk
  # )
  # 
  # # if (any(sapply(models, function(m) "fraglen" %in% m$offset))) {
  # #   ## -- fragment bias --
  # #   pos.count <- fragtypes.sub$count > 0
  # #   fraglens <- rep(fragtypes.sub$fraglen[pos.count], fragtypes.sub$count[pos.count])
  # #   fraglen.density <- density(fraglens)
  # #   fragtypes.sub$logdfraglen <- log(matchToDensity(fragtypes.sub$fraglen, fraglen.density))
  # #   # with(fragtypes.sub, plot(fraglen, exp(logdfraglen), cex=.1))
  # #   fitpar.sub[["fraglen.density"]] <- fraglen.density
  # # }
  # # 
  # # if (any(sapply(models, function(m) "vlmm" %in% m$offset))) {
  # #   ## -- random hexamer priming bias with VLMM --
  # #   pos.count <- fragtypes.sub$count > 0
  # #   fivep <- fragtypes.sub$fivep[fragtypes.sub$fivep.test & pos.count]
  # #   threep <- fragtypes.sub$threep[fragtypes.sub$threep.test & pos.count]
  # #   vlmm.fivep <- fitVLMM(fivep, gene.seqs)
  # #   vlmm.threep <- fitVLMM(threep, gene.seqs)
  # #   ## par(mfrow=c(2,1))
  # #   ## plotOrder0(vlmm.fivep$order0)
  # #   ## plotOrder0(vlmm.threep$order0)
  # #   
  # #   # now calculate log(bias) for each fragment based on the VLMM
  # #   fragtypes.sub <- addVLMMBias(fragtypes.sub, vlmm.fivep, vlmm.threep)
  # #   fitpar.sub[["vlmm.fivep"]] <- vlmm.fivep
  # #   fitpar.sub[["vlmm.threep"]] <- vlmm.threep
  # # }
  # 
  # # allow a gene-specific intercept (although mostly handled already with downsampling)
  # fragtypes.sub$gene <- factor(rep(seq_along(gene.nrows), gene.nrows))
  # for (modeltype in names(models)) {
  #   if (is.null(models[[modeltype]]$formula)) {
  #     next
  #   }
  #   # message("fitting model type: ",modeltype)
  #   f <- models[[modeltype]]$formula
  #   offset <- numeric(nrow(fragtypes.sub))
  #   # if ("fraglen" %in% models[[modeltype]]$offset) {
  #   #   # message("-- fragment length correction")
  #   #   offset <- offset + fragtypes.sub$logdfraglen
  #   # }
  #   # if ("vlmm" %in% models[[modeltype]]$offset) {
  #   #   # message("-- VLMM fragment start/end correction")
  #   #   offset <- offset + fragtypes.sub$fivep.bias + fragtypes.sub$threep.bias
  #   # }
  #   if (!all(is.finite(offset))) stop("offset needs to be finite")
  #   fragtypes.sub$offset <-  offset
  #   if ( speedglm ) {
  #     # mm.small <- sparse.model.matrix(f, data=fragtypes.sub)
  #     mm.small <- model.matrix(formula(f), data=fragtypes.sub)
  #     stopifnot(all(colSums(abs(mm.small)) > 0))
  #     fit <- speedglm.wfit(fragtypes.sub$count, mm.small,
  #                          family=poisson(), 
  #                          weights=fragtypes.sub$wts,
  #                          offset=fragtypes.sub$offset)
  #   } else {
  #     fit <- glm(formula(f),
  #                family=poisson,
  #                data=fragtypes.sub,
  #                weights=fragtypes.sub$wts,
  #                offset=fragtypes.sub$offset)
  #   }
  #   fitpar.sub[["coefs"]][[modeltype]] <- fit$coefficients
  #   fitpar.sub[["summary"]][[modeltype]] <- summary(fit)$coefficients
  # }
  # fitpar.sub
  fragtypes.sub.list
}
