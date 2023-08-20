import sys

from models.student_data import StudentData
from pipeline.predict_pipeline import PredictPipeline
from pipeline.transform_pipeline import TransformPipeline

from logger import logging
from exception import CustomException
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import RadioField, StringField, SubmitField, SelectField
from wtforms.validators import InputRequired, NumberRange, DataRequired
from flask_wtf import CSRFProtect

SECRET_KEY = 'development'

app = Flask(__name__)
app.secret_key = b'_53oi3uriq9pifpff;apl'
csrf = CSRFProtect(app)

class StudentForm(FlaskForm):
    subject = RadioField('Subject', choices = [('maths', 'Maths'), ('portugese', 'Portugese')], validators=[DataRequired("Subject value is required")]) 
    school = RadioField('School', choices = [('GP', 'Gabriel Pereira'), ('MS', 'Mousinho da Silveira')], validators=[DataRequired("School value is required")])
    sex = RadioField('Sex', choices=[('M', 'Male'), ('F', 'Female')], validators=[InputRequired("Sex value is required")])
    age = StringField('Age', validators=[NumberRange(min=15, max=22), InputRequired("Age is requires")])
    address = RadioField('Address', choices=[("U", "Urban"), ("R", "Rural")], validators=[InputRequired("Address required")])
    famsize = RadioField('Famsize', choices=[("LE3", "Less or equal to 3"), ("GT3", "Greater than 3")], validators=[InputRequired("Family size required")])
    pstatus = RadioField('Pstatus', choices=[("T", "Living Together"), ("A", "Apart")], validators=[InputRequired("Parent's cohabitation status required")])
    medu = SelectField('Medu', choices=[(0, 'None'), (1, 'Primary Education (4th grade)'), (2, '5th to 9th grade'), (3, 'Secondary education'),
                                        (4, 'Higher Education')], validators=[InputRequired("Mothers education required")])
    fedu = SelectField('Fedu', choices=[(0, 'None'), (1, 'Primary Education (4th grade)'), (2, '5th to 9th grade'), (3, 'Secondary education'),
                                        (4, 'Higher Education')], validators=[InputRequired("Fathers Education Required")])
    mjob = SelectField('Mjob', choices=[("teacher","teacher"), ("health", "health"), ("services", "services"),
                                        ("at_home", "at_home"), ("other", "other")], validators=[InputRequired("Mothers job required")])
    fjob = SelectField('Fjob', choices=[("teacher","teacher"), ("health", "health"), ("services", "services"),
                                        ("at_home", "at_home"), ("other", "other")], validators=[InputRequired("Fathers job required")])
    reason = SelectField('Reason', choices=[("home", "Home"), ("reputation", "Reputation"), ("course", "Course"), 
                                            ("other", "Other")], validators=[InputRequired("Reason for choosing this school is required")])
    guardian = SelectField('Guardian', choices=[('mother', 'Mother'), ('father', 'Father'), ('other', 'Other')], 
                           validators=[InputRequired("Student's Guardian is required")])
    traveltime = SelectField('TravelTime', choices=[(1, '<15 min'), (2, '15 to 30 min'), (3, '30 min. to 1 hour'), 
                                                    (4, '>1 hour')], validators=[InputRequired("Travel time from home to school is required")])
    studytime = SelectField('StudyTime', choices=[(1, '<2 hours'), (2, '2 to 5 hours'), (3, '5 to 10 hours'), 
                                                    (4, '>10 hours')], validators=[InputRequired("Weekly study time is required")])
    failures = StringField('Failures', validators=[NumberRange(min=0, max=3), InputRequired("Number of past class failures number required")])
    schoolsup = RadioField('Schoolsup', choices=[(0, 'Yes'), (1, 'No')], validators=[InputRequired('Extra educational support value required')])
    famsup =  RadioField('Famsup', choices=[(0, 'Yes'), (1, 'No')], validators=[InputRequired('Family educational support value required')])
    paid = RadioField('Paid', choices=[(0, 'Yes'), (1, 'No')], validators=[InputRequired('Extra paid classes value is required')])
    activities = RadioField('Activities', choices=[(0, 'Yes'), (1, 'No')], validators=[InputRequired('Extra curricular activites value is required')])
    nursery = RadioField('Nursery', choices=[(0, 'Yes'), (1, 'No')], validators=[InputRequired('Nursery value is required')])
    higher = RadioField('Higher', choices=[(0, 'Yes'), (1, 'No')], validators=[InputRequired('Higher value is required')])
    internet = RadioField('Internet', choices=[(0, 'Yes'), (1, 'No')], validators=[InputRequired('Internet value is required')])
    romantic = RadioField('Romantic', choices=[(0, 'Yes'), (1, 'No')], validators=[InputRequired('Romantic value is required')])
    famrel = StringField('Famrel', validators=[NumberRange(min=1, max=5), InputRequired("Quality of family relationship value is required")])
    freetime = StringField('Freetime', validators=[NumberRange(min=1, max=5), InputRequired("Free time after school value is required")])
    goout = StringField('Goout', validators=[NumberRange(min=1, max=5), InputRequired("Number of time going out with friends value is required")])
    dalc = StringField('Dalc', validators=[NumberRange(min=1, max=5), InputRequired("Workday alcohol consumption value is required")])
    walc = StringField('Walc', validators=[NumberRange(min=1, max=5), InputRequired("Weekend alcohol consumption value is required")])
    health = StringField('Health', validators=[NumberRange(min=1, max=5), InputRequired("Current health status value is required")])
    absences = StringField('Absences', validators=[NumberRange(min=0, max=93), InputRequired("Number of time absence in class is required")])
    g1 = StringField('G1', validators=[NumberRange(min=0, max=20), InputRequired("First period grade value is required")])
    g2 = StringField('G2', validators=[NumberRange(min=0, max=20), InputRequired("Second period grade value is required")])
    submit = SubmitField('Predict your performance')

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    form = StudentForm()
    results = None
    try:
        if request.method == 'POST':
            logging.info("Getting data from html form: (POST)")
            data = StudentData(
                str(form.school.data),
                str(form.sex.data),
                int(form.age.data),
                str(form.address.data),
                str(form.famsize.data),
                str(form.pstatus.data),
                int(form.medu.data or 0),
                int(form.fedu.data or 0),
                str(form.mjob.data),
                str(form.fjob.data),
                str(form.reason.data),
                str(form.guardian.data),
                int(form.traveltime.data or 0),
                int(form.studytime.data or 0),
                int(form.failures.data),
                int(form.schoolsup.data or 0),
                int(form.famsup.data or 0),
                int(form.paid.data or 0),
                int(form.activities.data or 0),
                int(form.nursery.data or 0),
                int(form.higher.data or 0),
                int(form.internet.data or 0),
                int(form.romantic.data or 0),
                int(form.famrel.data),
                int(form.freetime.data),
                int(form.goout.data),
                int(form.dalc.data),
                int(form.walc.data),
                int(form.health.data),
                int(form.absences.data),
                int(form.g1.data),
                int(form.g2.data)
            )
            df = data.convert_raw_data_to_data_frame()
            logging.info(f"Shape of dataframe is {df.shape}")
            transform_pipeline = TransformPipeline(df)
            transformed_df = transform_pipeline.transform_pipeline()
            predict_pipeline = PredictPipeline(transformed_df, str(form.subject.data))
            results = predict_pipeline.predict()
        return render_template('home.html', form=form, results=results)
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)