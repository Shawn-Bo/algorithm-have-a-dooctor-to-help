import typer
import job0_gen_data
import job1_train

app = typer.Typer()
app.add_typer(job0_gen_data.app, name="gen-data")
app.add_typer(job1_train.app, name="train")

@app.command()
def job2_test():
    pass


if __name__ == "__main__":
    app()
