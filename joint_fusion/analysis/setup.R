# native r + renv, project-local library. works with positron and rstudio.
# run once from this folder: Rscript setup.R  (or source in the console)
if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
renv::init(bare = TRUE)

renv::install("bioc::fgsea")
renv::install(c("msigdbr", "ggplot2", "rmarkdown", "knitr", "tinytex", "kableExtra"))

try(install.packages(
  "msigdbdf",
  repos = c("https://igordot.r-universe.dev", "https://cloud.r-project.org")
))

if (!tinytex::is_tinytex()) tinytex::install_tinytex()

renv::snapshot()

suppressMessages({library(fgsea); library(msigdbr)})
cat("fgsea", as.character(packageVersion("fgsea")),
    "| msigdbr", as.character(packageVersion("msigdbr")), "\n")
