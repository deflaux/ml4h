# mylibpaths <- .libPaths()
# mylibpaths <- c(mylibpaths, '/home/jupyter-user/packages/R/lib')
# .libPaths(mylibpaths)
# install.packages('/home/jupyter-user/src/GenABEL.data.tar.gz',
# 		 lib='/home/jupyter-user/packages/R/lib/', repos=NULL)
# install.packages('/home/jupyter-user/src/GenABEL.tar.gz',
# 		 lib='/home/jupyter-user/packages/R/lib/', repos = NULL)
# install.packages("BiocManager",
# 		 lib='/home/jupyter-user/packages/R/lib/', repos = 'http://cran.us.r-project.org')
# library("BiocManager")
# # Install Bioconductor packages
# BiocManager::install(c("snpStats", "SNPRelate", "rtracklayer", "biomaRt", "AnnotationDbi"),
# 		     lib='/home/jupyter-user/packages/R/lib/')
# install.packages(c('plyr', 'LDheatmap','doParallel', 'ggplot2', 'coin', 'igraph', 'devtools', 'downloader'),
# 		 lib='/home/jupyter-user/packages/R/lib/')
# library(devtools)
# install_url("http://cran.r-project.org/src/contrib/Archive/postgwas/postgwas_1.11.tar.gz",
# 	    lib='/home/jupyter-user/packages/R/lib/')


install.packages('/home/jupyter-user/src/GenABEL.data.tar.gz', repos=NULL)
install.packages('/home/jupyter-user/src/GenABEL.tar.gz', repos = NULL)
install.packages("BiocManager", repos = 'http://cran.us.r-project.org')
library("BiocManager")
BiocManager::install(c("snpStats", "SNPRelate", "rtracklayer", "biomaRt", "AnnotationDbi"))
install.packages(c('plyr', 'LDheatmap','doParallel', 'ggplot2', 'coin', 'igraph', 'devtools', 'downloader'))
library(devtools)
install_url("http://cran.r-project.org/src/contrib/Archive/postgwas/postgwas_1.11.tar.gz")
install.packages('IRkernel')
IRkernel::installspec(user = FALSE)

