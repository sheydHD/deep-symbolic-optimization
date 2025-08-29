"""Tools to evaluate generated logfiles based on log directory."""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# pkg_resources is deprecated, removed unused import
import re
import glob
import os

import click
import pandas as pd
import seaborn as sns
import json5 as json
from matplotlib import pyplot as plt

class LogEval():
    """Class to hold all logged information and provide tools
    to analyze experiments."""

    PLOT_HELPER = {
        "binding": {
            "name": "Binding Summary",
            "x_label": ["Epoch"] * 19,
            'y_label': [
                'Reward Best',
                'Reward Max',
                'Reward Avg Full',
                'Reward Avg Sub',
                'L Avg Full',
                'L Avg Sub',
                'EWMA',
                'Unique Full',
                'Unique Sub',
                'Novel Full',
                'Novel Sub',
                'A Avg Full',
                'A Avg Sub',
                'Invalid Avg Full',
                'Invalid Avg Sub',
                ],
            "x": ["index"] * 19,
            "y": [
                "r_best",
                "r_max",
                "r_avg_full",
                "r_avg_sub",
                "l_avg_full",
                "l_avg_sub",
                "ewma",
                "n_unique_full",
                "n_unique_sub",
                "n_novel_full",
                "n_novel_sub",
                "a_ent_full",
                "a_ent_sub",
                "invalid_avg_full",
                "invalid_avg_sub"
                ]
        },
        "hof": {
            "name": "Hall of Fame",
            "x_label": [
                "HoF reward distrubtion",
                "HoF error distrubtion",
                "HoF test reward distrubtion"],
            'y_label': [
                'Reward',
                'Error',
                'Test Reward'],
            "x": [
                "index",
                "index",
                "index"],
            "y": [
                "r",
                "nmse_test",
                "r_avg_test"]
        },
        "pf": {
            "name": "Pareto Front",
            "x_label": [
                "Complexity",
                "Complexity"],
            'y_label': [
                'Reward',
                'Error'],
            "x": [
                "complexity",
                "complexity"],
            "y": [
                "r",
                "nmse_test"]
        }
    }

    def __init__(self, config_path):
        """Load config, summary, hof, and pf."""
        print("-- LOADING LOGS START ----------------")
        self.warnings = []
        self.metrics = {}
        # Load config
        self.save_path = config_path
        self.config = self._get_config()
        # Load summary data (one row per seed)
        self.summary_df = self._get_summary()
        if self.summary_df is not None:
            print("Successfully loaded summary data")
        self.n_seeds = len(self.summary_df) if self.summary_df is not None else "N/A"
        # Load HOF
        self.hof_df = self._get_log(log_type="hof")
        if self.hof_df is not None:
            print("Successfully loaded Hall of Fame data")
        # Load pareto front
        self.pf_df = self._get_log(log_type="pf")
        if self.pf_df is not None:
            print("Successfully loaded Pareto Front data")
        # Show any warnings that occured during loading the data
        if len(self.warnings) > 0:
            print("*** WARNING:")
            for warning in self.warnings:
                print(f"    --> {warning}")
        print("-- LOADING LOGS END ------------------")

    def _get_config(self):
        """Read the experiment's config file."""

        with open(os.path.join(self.save_path, "config.json")) as f:
            config = json.load(f)

        return config

    def _get_summary(self):
        """Read summarized benchmark data for each seed."""

        summary_df = None
        try:
            summary_path = os.path.join(self.save_path,
                                        "summary.csv")
            summary_df = pd.read_csv(summary_path)
            summary_df = summary_df.reset_index(drop=True)
            summary_df.sort_values("seed")
            try:
                self.metrics["success_rate"] = summary_df["success"].mean()
            except:
                self.metrics["success_rate"] = 0.0
        except Exception as e:
            self.warnings.append(f"Can't load summary: {e}")

        return summary_df

    def _get_log(self, log_type):
        """Read data from log files ("hof" or "pf")."""

        log_dfs = []

        # Get files that match regexp
        task_name = self.config["experiment"]["task_name"]
        r = re.compile(fr"dso_{task_name}_\d+_{log_type}.csv")
        files = filter(r.match, os.listdir(self.save_path))
        files = [os.path.join(self.save_path, f) for f in files]
        seeds = [int(f.split("_")[-2]) for f in files]

        if len(files) == 0:
            self.warnings.append(f"No data for {log_type}!")
            return None

        # Load each df
        for f, seed in zip(files, seeds):
            df = pd.read_csv(f)
            df.insert(0, "seed", seed)
            log_dfs.append(df)

        # Combine them all
        log_df = pd.concat(log_dfs)

        # Sort HOF
        if log_type == "hof":
            if self.config["task"]["task_type"] == "binding":
                log_df = log_df.sort_values(by=["r"], ascending=False)
            else:
                log_df = log_df.sort_values(by=["r", "success", "seed"], ascending=False)

        # Compute PF across all runs
        if log_type == "pf":
            log_df = self._apply_pareto_filter(log_df)
            log_df = log_df.sort_values(by=["r", "complexity", "seed"], ascending=False)

        log_df = log_df.reset_index(drop=True)
        log_df["index"] = log_df.index

        return log_df

    def _apply_pareto_filter(self, df):
        """Filter out dominated solutions."""
        filtered_df = pd.DataFrame(columns=df.columns)
        for _, row in df.iterrows():
            if len(filtered_df) == 0:
                # For the first row, just add it
                filtered_df = pd.concat([filtered_df, pd.DataFrame([row])], ignore_index=True)
                continue

            r = row["r"]
            complexity = row["complexity"]

            # Check if solution is dominated
            dominated = False
            rows_to_remove = []
            for i, existing_row in filtered_df.iterrows():
                r_existing = existing_row["r"]
                complexity_existing = existing_row["complexity"]

                # Existing solution dominates new solution
                if r_existing >= r and complexity_existing <= complexity and (r_existing > r or complexity_existing < complexity):
                    dominated = True
                    break

                # New solution dominates existing solution
                if r >= r_existing and complexity <= complexity_existing and (r > r_existing or complexity < complexity_existing):
                    rows_to_remove.append(i)

            # Remove dominated existing solutions
            if rows_to_remove:
                filtered_df = filtered_df.drop(rows_to_remove)

            # Add new solution if not dominated
            if not dominated:
                filtered_df = pd.concat([filtered_df, pd.DataFrame([row])], ignore_index=True)

        return filtered_df

    def plot_results(self, results, log_type, boxplot_on=False, show_plots=False, save_plots=False):
        """Plot data from log files ("hof" or "pf")."""
        col_count = 0
        _x = []
        _y = []
        _x_label = []
        _y_label = []
        for i in range(len(self.PLOT_HELPER[log_type]["y"])):
            if self.PLOT_HELPER[log_type]["y"][i] in results:
                col_count += 1
                _x.append(self.PLOT_HELPER[log_type]["x"][i])
                _y.append(self.PLOT_HELPER[log_type]["y"][i])
                _x_label.append(self.PLOT_HELPER[log_type]["x_label"][i])
                _y_label.append(self.PLOT_HELPER[log_type]["y_label"][i])
        row_count = 2 if boxplot_on else 1
        if log_type == "binding":
            row_count = 5
            col_count = 4
        fig, ax = plt.subplots(row_count, col_count, squeeze=0, figsize=(8 * col_count, 4 * row_count))
        for i in range(col_count):
            if log_type == "binding":
                for row in range(row_count):
                    data_id = i + row * col_count
                    if data_id < len(_x):
                        sns.lineplot(data=results, x=_x[data_id], y=_y[data_id], ax=ax[row, i])
                        ax[row, i].set_xlabel(_x_label[data_id])
                        ax[row, i].set_ylabel(_y_label[data_id])
            else:
                if log_type == 'pf':
                    ax[0, i].scatter(results[_x[i]], results[_y[i]], s=75)
                else:
                    sns.lineplot(data=results, x=_x[i], y=_y[i], ax=ax[0, i])
                    if boxplot_on:
                        sns.boxplot(results[_y[i]], ax=ax[1, i])
                        ax[1, i].set_xlabel( _y[i])
                ax[0, i].set_xlabel(_x_label[i])
                ax[0, i].set_ylabel(_y_label[i])
        plt.suptitle(
            "{} - {}".format(self.PLOT_HELPER[log_type]["name"], self.config["experiment"]["task_name"]),
            fontsize=14)
        plt.tight_layout()
        if save_plots:
            save_path = os.path.join(self.save_path, "dso_{}_plot_{}.png".format(self.config["experiment"]["task_name"], log_type))
            print("  Saving {} plot to {}".format(self.PLOT_HELPER[log_type]["name"], save_path))
            plt.savefig(save_path)
        if show_plots:
            plt.show()
        plt.close()

    def analyze_log(self, show_count=5, show_hof=True, show_pf=True, show_plots=False, save_plots=False, show_binding=False):
        """Generates a summary of important experiment outcomes."""
        print("\n-- ANALYZING LOG START --------------")
        try:
            print(f"Task_____________{self.config['task']['task_type']}")
            print(f"Source path______{self.save_path}")
            print(f"Runs_____________{self.n_seeds}")
            print(f"Max Samples/run__{self.config['training']['n_samples']}")
            if "success_rate" in self.metrics:
                print(f"Success_rate_____{self.metrics['success_rate']}")
            if len(self.warnings) > 0:
                print("Found issues:")
                for warning in range(len(self.warnings)):
                    print(f"  {warning}")
            if self.hof_df is not None and show_hof:
                hof_show_count = min(show_count, len(self.hof_df))
                print(f'Hall of Fame (Top {hof_show_count} of {len(self.hof_df)})____')
                for i in range(hof_show_count):
                    print('  {:3d}: S={:03d} R={:8.6f} <-- {}'.format(
                        i, self.hof_df.iloc[i]['seed'], self.hof_df.iloc[i]['r'],
                        self.hof_df.iloc[i]['expression']))
                if show_plots or save_plots:
                    self.plot_results(
                        self.hof_df, log_type="hof", boxplot_on=True,
                        show_plots=show_plots, save_plots=save_plots)
            if self.pf_df is not None and show_pf:
                print(f'Pareto Front ({min(show_count,len(self.pf_df.index))} of {len(self.pf_df.index)})____')
                for i in range(min(show_count,len(self.pf_df.index))):
                    print('  {:3d}: S={:03d} R={:8.6f} C={:.2f} <-- {}'.format(
                        i, self.pf_df.iloc[i]['seed'], self.pf_df.iloc[i]['r'],
                        self.pf_df.iloc[i]['complexity'], self.pf_df.iloc[i]['expression']))
                if show_plots or save_plots:
                    self.plot_results(
                        self.pf_df, log_type="pf",
                        show_plots=show_plots, save_plots=save_plots)
        except FloatingPointError:
            print("Error when analyzing!")
            for warning in self.warnings:
                print(f"    --> {warning}")
        print("-- ANALYZING LOG END ----------------")


@click.command()
@click.argument('config_path', default=None, type=str)
@click.option('--show_count', default=10, type=int, help="Number of results we want to see from each metric.")
@click.option('--show_hof', is_flag=True, help='Show Hall of Fame results.')
@click.option('--show_pf', is_flag=True, help='Show Pareto Front results.')
@click.option('--show_plots', is_flag=True, help='Generate plots and show results as simple plots.')
@click.option('--save_plots', is_flag=True, help='Generate plots and safe to log file as simple plots.')
def main(config_path, show_count, show_hof, show_pf, show_plots, save_plots):

    log = LogEval(config_path)
    log.analyze_log(
        show_count=show_count,
        show_hof=show_hof,
        show_pf=show_pf,
        show_plots=show_plots,
        save_plots=save_plots)


if __name__ == "__main__":
    main()
