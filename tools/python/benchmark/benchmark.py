import subprocess
import sys
import click
import os

@click.command()
@click.argument('config_file')
@click.option('--benchmark', default=None, help='Specify a benchmark to run.')
def main(config_file, benchmark):
    """Runs a benchmark using dso.run."""
    click.echo(f"Running benchmark with config: {config_file}")
    
    # Use relative path to .venv from project root
    project_root = click.get_current_context().find_root().info.get('project_root', os.getcwd())
    python_executable = os.path.join(project_root, ".venv", "bin", "python")

    cmd = [python_executable, "-m", "dso.run", config_file]
    if benchmark:
        cmd.extend(["--benchmark", benchmark])
    
    # Set PYTHONPATH to the project root for module discovery
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root

    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError:
        click.echo(click.style("dso or python not found. Have you run setup?", fg="red"), err=True)
        sys.exit(1)
    except subprocess.CalledProcessError:
        click.echo(click.style("Benchmark failed.", fg="red"), err=True)
        sys.exit(1)
    click.echo(click.style("Benchmark finished.", fg="green"))

if __name__ == "__main__":
    main()
