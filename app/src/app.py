from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
import uvicorn
import argparse
import logging
from model import evaluate
from model_rec import predict
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("start_form.html",
                                      {"request": request,
                                       "message": 'lets start!'})


@app.post("/recognize")
def process_request(file: UploadFile, request: Request):
    """save file to the local folder and send the image to the process function"""
    save_pth = "tmp/" + file.filename
    app_logger.info(f'processing file - detection {save_pth}')
    with open(save_pth, "wb") as fid:
        fid.write(file.file.read())
    res, res_path, c, f_p = evaluate(save_pth)
    if res == 'OK' and c != 0:
        app_logger.info(f'detected {c} faces')
        status, simil = predict(f_p)
        message = f"Detected {c} faces. Difference Coefficient between faces - {simil}"
        return templates.TemplateResponse("rec_form.html",
                                          {"request": request,
                                           "res": res,
                                           "message": message, "path": res_path})
    elif res == 'OK' and c == 0:
        app_logger.info(f'detected {c} faces')
        message = f'detected {c} faces'
        return templates.TemplateResponse("start_form.html",
                                          {"request": request,
                                           "message": message})
    else:
        app_logger.warning(f'some problems {res}')
        return templates.TemplateResponse("error_form.html",
                                          {"request": request,
                                           "result": res,
                                           "name": file.filename})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
