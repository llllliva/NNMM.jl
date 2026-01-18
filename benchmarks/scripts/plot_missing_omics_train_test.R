#!/usr/bin/env Rscript
# Generate plots for the missing-omics train/test benchmark.
#
# Inputs:
#   - benchmarks/missing_omics_train_test_results.csv
#
# Outputs:
#   - benchmarks/missing_omics_train_test_plot.png
#
# This is a base-R fallback for environments without Python/matplotlib.

`%||%` <- function(a, b) if (!is.null(a) && nzchar(a)) a else b

# Locate the script directory (works when invoked via Rscript).
full_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", full_args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[1]) else ""
here <- if (nzchar(script_path)) dirname(normalizePath(script_path)) else normalizePath("benchmarks", mustWork = FALSE)
results_csv <- file.path(here, "missing_omics_train_test_results.csv")
out_png <- file.path(here, "missing_omics_train_test_plot.png")

if (!file.exists(results_csv)) {
  stop(paste0("Missing input CSV: ", results_csv, "\nRun the benchmark first to generate it."))
}

df <- read.csv(results_csv, stringsAsFactors = FALSE)

num_cols <- c(
  "train_missing_pct",
  "test_missing_pct",
  "ebv_test_total",
  "ebv_test_indirect",
  "epv_test_total",
  "epv_test_trait",
  "time_seconds"
)
for (k in num_cols) {
  if (k %in% names(df)) df[[k]] <- as.numeric(df[[k]])
}

seed <- if ("seed" %in% names(df)) df$seed[1] else NA
chain_length <- if ("chain_length" %in% names(df)) df$chain_length[1] else NA
burnin <- if ("burnin" %in% names(df)) df$burnin[1] else NA
missing_mode <- if ("missing_mode" %in% names(df)) df$missing_mode[1] else ""

groups <- split(df, df$test_missing_pct)
groups <- lapply(groups, function(g) g[order(g$train_missing_pct), , drop = FALSE])

colors <- c(`0` = "#1f77b4", `1` = "#d62728")
labels <- c(`0` = "Test omics 0% missing", `1` = "Test omics 100% missing")

series <- function(test_missing, key) {
  g <- groups[[as.character(test_missing)]]
  if (is.null(g) || nrow(g) == 0) return(list(x = numeric(0), y = numeric(0)))
  xs <- g$train_missing_pct * 100.0
  ys <- g[[key]]
  ys[is.nan(ys)] <- NA
  list(x = xs, y = ys)
}

plot_metric <- function(key, title) {
  plot(
    NA,
    xlim = c(0, 100),
    ylim = c(0, 1),
    xlab = "",
    ylab = "",
    main = title
  )
  grid(col = "gray90")
  for (tm in sort(unique(df$test_missing_pct))) {
    s <- series(tm, key)
    lines(s$x, s$y, type = "o", pch = 16, cex = 0.7, lwd = 2, col = colors[as.character(tm)] %||% "black")
  }
}

png(out_png, width = 1100, height = 800, res = 150)

layout(matrix(c(1, 2, 3, 4, 5, 5), nrow = 3, byrow = TRUE), heights = c(1, 1, 0.8))
par(mar = c(4, 4, 2.5, 1.5), oma = c(0, 0, 3, 0))

plot_metric("ebv_test_total", "EBV (test) vs genetic_total")
legend(
  "bottomleft",
  legend = labels[names(labels) %in% names(groups)],
  col = colors[names(colors) %in% names(groups)],
  lwd = 2,
  pch = 16,
  cex = 0.8,
  bty = "n"
)

plot_metric("ebv_test_indirect", "EBV (test) vs genetic_indirect")
plot_metric("epv_test_total", "EPV (test) vs genetic_total")
plot_metric("epv_test_trait", "EPV (test) vs trait1")

# Runtime plot (spanning both columns)
plot(
  NA,
  xlim = c(0, 100),
  ylim = range(df$time_seconds, finite = TRUE),
  xlab = "Training omics missing (%)",
  ylab = "Seconds",
  main = "Runtime (seconds) per configuration"
)
grid(col = "gray90")
for (tm in sort(unique(df$test_missing_pct))) {
  s <- series(tm, "time_seconds")
  lines(s$x, s$y, type = "o", pch = 16, cex = 0.7, lwd = 2, col = colors[as.character(tm)] %||% "black")
}

mtext(
  sprintf(
    "Missing Omics Train/Test Benchmark (seed=%s, chain=%s, burnin=%s, mode=%s)",
    seed, chain_length, burnin, missing_mode
  ),
  outer = TRUE,
  cex = 1.1
)

dev.off()

cat("Wrote: ", out_png, "\n", sep = "")
