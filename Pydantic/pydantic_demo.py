from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Padma'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(
        gt=0, 
        lt=10, 
        default=5, 
        description='A decimal value representing the cgpa of the student'
    )

new_student = {
    'age': '26',
    'email': 'abc@gmail.com',
    'cgpa': 50
}

student = Student(**new_student)
print(student.age, student.cgpa, student.email)
student_dict = dict(student)

print(student_dict['age'])
student_json = student.model_dump_json()
print(student_json)
# print(student_json['age'])

import os
os.makedirs("output", exist_ok=True)
with open("output/student_data.json", "w") as f:
    f.write(student_json)
