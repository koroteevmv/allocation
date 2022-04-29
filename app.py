import logging
import os
import pickle
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, redirect, send_file
from werkzeug.utils import secure_filename

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

from Model import Model

app = Flask(__name__)


@app.route('/')
def home():
    res = [{
        'name': f,
        'date': datetime.utcfromtimestamp(
            os.path.getmtime(os.path.join('./models', f))
        ).strftime('%Y-%m-%d %H:%M:%S')
    } for f in os.listdir('./models') if os.path.isfile(os.path.join('./models', f))]
    log.debug(res)
    return render_template('model_list.html', models=res)


@app.route('/add_model', methods=['GET', 'POST'])
def add_model():
    if request.method == 'POST':
        log.debug(request.form["model_name"])
        filename = os.path.join('models', request.form["model_name"])
        model = Model()
        if os.path.isfile(os.path.join('models', request.form["model_name"])):
            return render_template('add_model.html', alert="Выберите другое имя, такой файл уже существует!")
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        return redirect(f'/')
    return render_template('add_model.html')


@app.route('/models/<name>/', methods=['GET', 'POST'])
def model(name):
    filename = os.path.join('models', name)
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    log.debug(model.stuff_tag)
    stuff = ''
    hours_stuff = 0
    num_stuff = 0
    courses = ''
    hours_courses = 0
    num_courses = 0

    try:
        stuff = model.stuff_hours.style.format('<span>{}<span>').render()
        hours_stuff = model.stuff_hours.opt.sum()
        num_stuff = len(model.list_stuff)
    except AttributeError:
        stuff = ''
        hours_stuff = 0
        num_stuff = 0
    try:
        courses = model.method_hours.style.format('<span>{}<span>').render()
        hours_courses = model.method_hours.hours.sum()
        num_courses = len(model.list_courses)
    except AttributeError:
        courses = ''
        hours_courses = 0
        num_courses = 0
    try:
        tags = len(model.list_tags)
    except:
        tags = 0
    try:
        matrix = model.stuff_tag.iloc[:, 0:15].style.format('<span>{}<span>').render()
    except AttributeError:
        matrix = ''

    return render_template('model_main.html',
                           stuff=stuff, courses=courses, tags=tags, matrix=matrix, name=name,
                           hours_stuff=hours_stuff,
                           hours_courses=hours_courses,
                           num_stuff=num_stuff,
                           num_courses=num_courses,
                           )


@app.route('/models/<name>/dwn/<component>')
def dwn(name, component):
    path = "./tmp/" + component + ".xls"
    filename = os.path.join('models', name)
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    res = None

    if component == "stuff":
        res = model.stuff_hours
    elif component == "courses":
        res = model.method_hours
    elif component == "tags":
        res = model.courses_tags
    elif component == "matrix":
        res = model.stuff_tag
    elif component == "stuff_tags_tmpl":
        res = pd.DataFrame(
            index=model.list_stuff,
            columns=model.list_tags,
        ).fillna(0.5)
        log.debug(res)
    elif component == "solution":
        res = model.result

    res.to_excel(path)

    return send_file(path, as_attachment=True)


@app.route('/models/<name>/edit/courses', methods=['GET', 'POST'])
def courses(name):
    filename = os.path.join('models', name)
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    if request.method == 'POST':
        log.debug(request.files)
        if 'courseFile' in request.files and request.files['courseFile']:
            file = request.files['courseFile']
            filename = secure_filename(file.filename)
            file.save(os.path.join('./tmp', filename))
            model.read_courses_hours(os.path.join('./tmp', filename))

        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        return redirect(request.url)

    try:
        courses = model.method_hours.to_html(index=True, table_id="T_my_id", classes='table table-striped', border=0)
    except AttributeError:
        courses = ''

    return render_template('courses.html',
                           courses=courses, name=name,
                           )


@app.route('/models/<name>/edit/stuff', methods=['GET', 'POST'])
def stuff(name):
    filename = os.path.join('models', name)
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    if request.method == 'POST':
        log.debug(request.files)
        if 'stuffFile' in request.files and request.files['stuffFile']:
            file = request.files['stuffFile']
            filename = secure_filename(file.filename)
            file.save(os.path.join('./tmp', filename))
            model.read_stuff_hours(os.path.join('./tmp', filename))

        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        return redirect(request.url)

    try:
        stuff = model.stuff_hours.to_html(index=True, table_id="T_my_id", classes='table table-striped', border=0)
    except AttributeError:
        stuff = ''

    return render_template('stuff.html',
                           stuff=stuff, name=name,
                           )


@app.route('/models/<name>/edit/stuff_tags', methods=['GET', 'POST'])
def stuff_tags(name):
    filename = os.path.join('models', name)
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    if request.method == 'POST':
        log.debug(request.files)
        if 'stuffFile' in request.files and request.files['stuffFile']:
            file = request.files['stuffFile']
            filename = secure_filename(file.filename)
            file.save(os.path.join('./tmp', filename))
            try:
                res = pd.read_csv(os.path.join('./tmp', filename), index_col=0).fillna(0.0).astype(
                    'float64').sort_index()
            except:
                res = pd.read_excel(os.path.join('./tmp', filename), index_col=0).fillna(0.0).astype(
                    'float64').sort_index()
            res = res[~res.index.duplicated(keep='last')]

            if not model.list_stuff.equals(res.index.drop_duplicates()):
                # log.warning("Список преподавателей отличается.")
                one, two = set(model.list_stuff), set(res.index.drop_duplicates())
                # log.warning("В модели отсутствует информация о следующих преподавателях:")
                [log.warning(f"\t{x}") for x in sorted(two.difference(one))]
                # log.warning("Их нагрузка не будет учтена.\n")
                # log.warning("В предпочтениях отсутствует информация о следующих преподавателях:")
                [log.warning(f"\t{x}") for x in sorted(one.difference(two))]
                # log.warning("Их предпочтения не будут учтены.")
                left = sorted(two.difference(one))
                right = sorted(one.difference(two))
                msg = """При формировании матрицы предпочтений преподавателей обнаружено несоответствие.
                    Обычно такое случается из-за 
                    опечаток или разного написания имен преподавателей. В таком случае рекомендуем исправить 
                    соответствующие названия в исходных файлах и перезагрузить таблицы нагрузки по преподавателям 
                    и\или предпочтений. Либо же это могут быть  новые преподаватели. Тогда рекомендуем собрать 
                    по ним предпочтения. Важнее, чтобы левая колонка оказалась пуста."""
                r_msg = "Преподаватели, которые есть в нагрузке, но отсутствуют в предпочтениях. Если продолжить, " \
                        "то нагрузка на них будет распределена случайным образом."
                l_msg = "Преподаватели, которые есть в предпочтениях, но отсутствуют в нагрузке. Если продолжить, " \
                        "то на них не будет распределена нагрузка."
                link = f"/models/{name}/stuff_tags_anyway/{filename}/"
                return render_template('identity_error.html',
                                       left=left, right=right, name=name,
                                       msg=msg, l_msg=l_msg, r_msg=r_msg, link=link
                                       )

            # model.read_stuff_tag(os.path.join('./tmp', filename))

        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        return redirect(request.url)

    log.debug(model.stuff_tag)

    try:
        matrix = model.stuff_tag.iloc[:, :10].to_html(index=True, table_id="T_my_id", classes='table table-striped',
                                                      border=0)
    except AttributeError:
        matrix = ''

    return render_template('stuff_tags.html',
                           matrix=matrix, name=name,
                           )


@app.route('/models/<name>/stuff_tags_anyway/<filename>/', methods=['GET', 'POST'])
def stuff_tags_anyway(name, filename):
    filename_ = os.path.join('models', name)
    with open(filename_, 'rb') as f:
        model = pickle.load(f)

    model.read_stuff_tag(os.path.join('./tmp', filename))

    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    return redirect(f'/models/{name}')


@app.route('/models/<name>/results')
def results(name):
    filename = os.path.join('models', name)
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    data = model.result.drop([
        "real", "opt", "stuff",
    ], axis=1).sort_index().to_html(index=False, table_id="T_my_id", classes='table table-striped', border=0,
                                    float_format='%10.2f')

    data3 = model.result.drop([
        "stuff", "hours",
        # "real", "opt", "stuff",
    ], axis=1).groupby(['name']).mean().sort_index().to_html(index=True, table_id="T_my_id2",
                                                             classes='table table-striped', border=0,
                                                             float_format='%10.2f')

    return render_template('results.html',
                           name=name,
                           data=data,
                           data3=data3,
                           )


@app.route('/models/<name>/generate')
def generate(name):
    filename = os.path.join('models', name)
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    tags = list(model.list_tags)
    courses = list(model.list_courses)
    if tags != courses:
        one, two = set(tags), set(courses)
        left = sorted(two.difference(one))
        right = sorted(one.difference(two))
        msg = """При формировании матрицы соответствия преподавателей дисциплинам обнаружено несоответствие 
        списка дисциплин по нагрузке и списка дисциплин по компетенциям. обычно такое может случиться из-за 
        опечаток или разного написания названий дисциплин. В таком случае рекомендуем исправить соответствующие
        названия в исходных файлах и перезагрузить таблицы дисциплин и\или предпочтений. Либо же это могут быть 
        новые дисциплины. Тогда рекомендуем собрать по ним предпочтения. Важнее, чтобы левая колонка оказалась пуста."""
        l_msg = "Дисциплины, которые есть в нагрузке, но отсутствуют в предпочтениях. Если продолжить, то они будут распределены случайным образом."
        r_msg = "Дисциплины, которые есть в предпочтениях, но отсутствуют в нагрузке. Если продолжить, они не будут распределены."
        link = f"/models/{name}/generate_anyway"
        return render_template('identity_error.html',
                               left=left, right=right, name=name,
                               msg=msg, l_msg=l_msg, r_msg=r_msg, link=link
                               )

    model.calc_courses_tags()

    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    return redirect(f"/models/{name}")


@app.route('/models/<name>/generate_anyway')
def generate_anyway(name):
    filename = os.path.join('models', name)
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    model.calc_courses_tags()

    # with open(filename, 'wb') as f:
    #     pickle.dump(model, f)
    return redirect(f'/models/{name}')


@app.route('/models/<name>/edit/tags')
def tags(name):
    filename = os.path.join('models', name)
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    model.calc_courses_tags()

    # with open(filename, 'wb') as f:
    #     pickle.dump(model, f)
    return render_template('tags.html',
                           name=name,
                           )


@app.route('/models/<name>/find_stuff')
def find_stuff(name):
    filename = os.path.join('models', name)
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    res = model.stuff_tag.dot(model.courses_tags.T).T.iloc[:, :].div(model.courses_tags.T.sum(axis=0), axis=0)
    res = res.melt(ignore_index=False)
    res = res.reset_index()
    res = res[res.value > 0]
    res.columns = ["Преподаватель", "Дисциплина", "Соответствие"]
    res = res[["Дисциплина", "Преподаватель", "Соответствие"]]
    res = res.to_html(index=False, table_id="T_my_id2",
                      classes='table table-striped', border=0,
                      float_format='{0:.0%}'.format)
    return render_template('find_stuff.html',
                           name=name,
                           data=res,
                           )


@app.route('/models/<name>/head_hunt')
def head_hunt(name):
    filename = os.path.join('models', name)
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    # log.debug(model.list_tags)
    res = pd.DataFrame(index=model.list_tags.tag, columns=["Компетенция", "Требуется", "Есть", "Дефицит"])
    res["Компетенция"] = res.index
    res["Требуется"] = (model.courses_tags.T.dot(model.method_hours.groupby("course").sum()))['hours']
    # log.debug(model.stuff_tag)
    # log.debug(model.stuff_hours)
    # log.debug(model.stuff_tag.T.dot(model.stuff_hours))
    res["Есть"] = (model.stuff_tag.T.dot(model.stuff_hours))['opt']
    res["Дефицит"] = res["Есть"] - res["Требуется"]
    # log.debug(res)
    res = res.to_html(index=False, table_id="T_my_id2",
                      classes='table table-striped', border=0,
                      float_format='%10.2f')
    return render_template('head_hunt.html',
                           name=name,
                           data=res,
                           )


# TODO pfuheprf ghjikjujlytq vjltkb

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
