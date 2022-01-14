use std::path::PathBuf;

use eyre::Result;
use plotters;
use plotters::prelude::*;

use extended_isolation_forest::{Forest, ForestOptions};

fn read_acceleration_data(
    filename: &str,
    apply_sliding_window: Option<usize>,
) -> eyre::Result<Vec<[f64; 3]>> {
    let file_path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "data", "acceleration", filename]
        .iter()
        .collect();

    let mut csv_reader = csv::Reader::from_path(file_path.as_path())?;
    let mut rows = Vec::new();
    for record_res in csv_reader.records() {
        let record = record_res?;
        rows.push([
            record[1].parse()?, // accel x
            record[2].parse()?, // accel y
            record[3].parse()?, // accel z
        ]);
    }

    if let Some(window_width) = apply_sliding_window {
        let mut smoothend = Vec::new();
        for window in rows.windows(window_width) {
            smoothend.push([
                window.iter().map(|v| v[0]).sum::<f64>() / window.len() as f64,
                window.iter().map(|v| v[1]).sum::<f64>() / window.len() as f64,
                window.iter().map(|v| v[2]).sum::<f64>() / window.len() as f64,
            ])
        }
        Ok(smoothend)
    } else {
        Ok(rows)
    }
}

const OUT_FILE_NAME: &str = "walking_stairs.png";

fn main() -> Result<()> {
    let smoothing = Some(20);

    // train a forest
    let forest = Forest::from_slice(
        read_acceleration_data("walking-stairs.train.csv", smoothing)?.as_slice(),
        &ForestOptions {
            n_trees: 100,
            sample_size: 600,
            max_tree_depth: None,
            extension_level: 1,
        },
    )?;

    let rows = read_acceleration_data("walking-stairs.annomaly.csv", smoothing)?;

    // plot results
    let root = BitMapBackend::new(OUT_FILE_NAME, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(10, 10, 10, 10);

    let (upper, lower) = root.split_vertically(500);
    let max_value = 3.0f64;

    // plot the accelerations
    let mut upper_chart = ChartBuilder::on(&upper)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 30)
        .caption("Acceleration while walking stairs", ("sans-serif", 14))
        .build_cartesian_2d(
            0.0..(rows.len() as f64 - 1.0),
            (max_value * -1.0)..max_value,
        )?;

    upper_chart
        .configure_mesh()
        //.disable_x_mesh()
        //.disable_y_mesh()
        .y_desc("Linear Acceleration (m/s²)")
        .x_desc("Time step")
        .draw()?;

    let line_styles = [
        (0, &BLUE, "Accel(z)"),
        (1, &GREEN, "Accel(y)"),
        (2, &RED, "Accel(z)"),
        //(3, &BLACK, "Accel(absolute)"),
    ];
    for (array_idx, style, label) in line_styles {
        upper_chart
            .draw_series(LineSeries::new(
                rows.iter()
                    .enumerate()
                    .map(|(idx, values)| (idx as f64, values[array_idx])),
                style.clone(),
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
    }

    upper_chart
        .configure_series_labels()
        .background_style(&RGBColor(200, 200, 200))
        .draw()?;

    let mut lower_chart = ChartBuilder::on(&lower)
        .margin_top(10)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 30)
        //.caption("Anomaly score", ("sans-serif", 14))
        .build_cartesian_2d(0.0..(rows.len() as f64 - 1.0), 0.3..1.0)?;

    lower_chart
        .configure_mesh()
        //.disable_x_mesh()
        //.disable_y_mesh()
        .y_desc("Anomaly score")
        .x_desc("Time step")
        .draw()?;

    // plot the detected anomalies
    lower_chart.draw_series(
        AreaSeries::new(
            rows.iter()
                .enumerate()
                .map(|(idx, row)| (idx as f64, forest.score(row))),
            0.0,
            &RED.mix(0.2),
        )
        .border_style(&RED),
    )?;

    root.present()?;
    println!("Result has been saved to {}", OUT_FILE_NAME);
    Ok(())
}
