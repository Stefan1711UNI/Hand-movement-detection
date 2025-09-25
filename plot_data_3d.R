library(plotly); library(dplyr); library(readr); library(viridis); library(htmlwidgets)

# find newest CSV in logs/
files <- list.files("logs", pattern = "\\.csv$", full.names = TRUE)
stopifnot(length(files) > 0)
csv_path <- files[which.max(file.info(files)$mtime)]
cat("Plotting from:", csv_path, "\n")

# read CSV
df <- read_csv(csv_path, show_col_types = FALSE)

# keep only raw data: cast to numeric and sort by time
df <- df %>%
  mutate(
    x_m = as.numeric(x_m),
    y_m = as.numeric(y_m),
    z_m = as.numeric(z_m),
    timestamp_ms = as.numeric(timestamp_ms)
  ) %>%
  arrange(timestamp_ms) %>%
  mutate(
    time_s = (timestamp_ms - min(timestamp_ms, na.rm = TRUE)) / 1000,
    idx = row_number()
  )


if (nrow(df) == 0) stop("CSV loaded but contains 0 rows")
if (sum(!is.na(df$x_m)) < 2) stop("Not enough numeric x_m points to plot. Check CSV columns and formatting.")

# aspect ratio so axes are visually comparable
xr <- range(df$x_m, na.rm = TRUE); yr <- range(df$y_m, na.rm = TRUE); zr <- range(df$z_m, na.rm = TRUE)
dx <- diff(xr); dy <- diff(yr); dz <- diff(zr); dmax <- max(c(dx,dy,dz,1e-6))
aspect <- list(x = dx/dmax + 0.01, y = dy/dmax + 0.01, z = dz/dmax + 0.01)

# build interactive 3D trace from data
p <- plot_ly(df,
             x = ~x_m, y = ~y_m, z = ~z_m,
             color = ~time_s, colors = viridis(200),
             type = "scatter3d",
             mode = "lines+markers",
             marker = list(size = 3),
             line = list(width = 3),
             text = ~paste("t(s):", round(time_s,3), "<br>idx:", idx)) %>%
  colorbar(title = "time (s)") %>%
  layout(scene = list(aspectmode = "manual", aspectratio = aspect),
         title = paste0("3D Wrist Trajectory — ", basename(csv_path)))

# print to Viewer
print(p)

# Save an HTML snapshot
out_html <- file.path("plots", paste0("trajectory_raw_", format(Sys.time(), "%Y-%m-%d_%H-%M-%S"), ".html"))
saveWidget(p, file = out_html, selfcontained = TRUE)
cat("Saved HTML to:", out_html, "\n")


