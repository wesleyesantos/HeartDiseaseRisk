

def sex_to_numeric(gender):
    if gender == "Female":
        return 0
    if gender == "Male":
        return 1
    
def age_to_numeric(age):
    if age == "18-24":
        return 0
    if age == "25-29":
        return 1
    if age == "30-34":
        return 2
    if age == "35-39":
        return 3
    if age == "40-44":
        return 4
    if age == "45-49":
        return 5
    if age == "50-54":
        return 6
    if age == "55-59":
        return 7
    if age == "60-64":
        return 8
    if age == "65-69":
        return 9
    if age == "70-74":
        return 10
    if age == "75-79":
        return 11
    if age == "80 or older":
        return 12
    
def BMI_to_numeric(bmi):
    if bmi == "Underweight (BMI < 18.5)":
        return 3
    if bmi == "Normal weight (18.5 <= BMI < 25.0)":
        return 0
    if bmi == "Overweight (25.0 <= BMI < 30.0)":
        return 2
    if bmi == "Obese (30.0 <= BMI < +Inf)":
        return 1
    
def gen_health_to_numeric(health):
    if health == "Excellent":
        return 0
    if health == "Fair":
        return 1
    if health == "Good":
        return 2
    if health == "Poor":
        return 3
    if health == "Very good":
        return 4
    
def diabetic_to_numeric(diabetic):
    if diabetic == "No":
        return 0
    if diabetic == "No, borderline diabetes":
        return 1
    if diabetic == "Yes":
        return 2
    if diabetic == "Yes (during pregnancy)":
        return 3

def alcohol_to_numeric(drinking):
    if drinking == "No":
        return 0
    if drinking == "Yes":
        return 1
    
def smoking_to_numeric(smoking):
    if smoking == "No":
        return 0
    if smoking == "Yes":
        return 1
    
def stroke_to_numeric(stroke):
    if stroke == "No":
        return 0
    if stroke == "Yes":
        return 1
    
def diffwalk_to_numeric(walking):
    if walking == "No":
        return 0
    if walking == "Yes":
        return 1
    
    
