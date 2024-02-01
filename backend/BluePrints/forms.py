import wtforms
from wtforms.validators import DataRequired, Email, EqualTo, Length
from models import User
from utils.redis_captcha import redis_get
import re


def check_email(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.fullmatch(regex, email):
        return True
    else:
        return False


class RegisterForm(wtforms.Form):
    username = wtforms.StringField(validators=[Length(min=2, max=25, message="Username format error!")])
    password = wtforms.StringField(validators=[Length(min=6, max=16, message="Password format error!")])
    password_confirm = wtforms.StringField(validators=[EqualTo("password", message="The two passwords do not match!")])
    email = wtforms.StringField(validators=[Email(message="Email format error!")])
    captcha = wtforms.StringField(validators=[DataRequired(message="The verification code cannot be empty!"),
                                              Length(min=4, max=4, message="Verification code length error!")])

    def validate_captcha(self, field):
        email = self.email.data
        captcha = self.captcha.data
        redis_captcha = redis_get(email)
        if not redis_captcha or captcha.lower() != redis_captcha.lower():
            raise wtforms.ValidationError("Email verification code error!")

    @staticmethod
    def validate_username(self, field):
        username = field.data
        user = User.query.filter_by(username=username).first()
        if user:
            raise wtforms.ValidationError("The username already exists!")

    @staticmethod
    def validate_email(self, field):
        email = field.data
        user = User.query.filter_by(email=email).first()
        if user:
            raise wtforms.ValidationError("The email already exists!")


class LoginForm(wtforms.Form):
    account = wtforms.StringField(validators=[Length(min=2, max=25, message="Account format error!")])
    password = wtforms.StringField(validators=[Length(min=6, max=16, message="Password format error!")])

    @staticmethod
    def validate_account(self, field):
        account = field.data
        if check_email(account):
            user = User.query.filter_by(email=account).first()
            if not user:
                raise wtforms.ValidationError("The email does not exist!")
        else:
            user = User.query.filter_by(username=account).first()
            if not user:
                raise wtforms.ValidationError("The username does not exist!")
