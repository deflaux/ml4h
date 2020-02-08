install.packages('/home/jupyter-user/src/GenABEL.data.tar.gz', repos = NULL)
install.packages('/home/jupyter-user/src/GenABEL.tar.gz', repos = NULL) 
# Install Bioconductor packages
BiocManager::install(c("snpStats", "SNPRelate", "rtracklayer", "biomaRt", "AnnotationDbi"))
install.packages(c('plyr', 'LDheatmap','doParallel', 'ggplot2', 'coin', 'igraph', 'devtools', 'downloader'))
library(devtools)
install_url("http://cran.r-project.org/src/contrib/Archive/postgwas/postgwas_1.11.tar.gz")