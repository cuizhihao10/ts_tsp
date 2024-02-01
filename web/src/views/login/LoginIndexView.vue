<template>
    <body>
        <div class="base">
            <!-- 注册登录界面 -->
            <div class="loginAndRegist">
                <!--登录表单-->
                <div class="loginArea">
                    <transition name="animate__animated animate__bounce" enter-active-class="animate__fadeInUp"
                        leave-active-class="animate__zoomOut" appear>
                        <!-- 标语 -->
                        <div v-show="isShow" class="title">
                            LOGIN
                        </div>
                    </transition>
                    <transition name="animate__animated animate__bounce" enter-active-class="animate__fadeInUp"
                        leave-active-class="animate__zoomOut" appear>
                        <!-- 密码框和用户名框 -->
                        <div v-show="isShow" class="pwdArea">
                            <el-form class="login_form" ref="loginUserRef" :model="loginUser" :rules="login_rules"
                                :label-position="labelPosition">
                                <el-form-item class="login_input" prop="account">
                                    <span style="font-size: 20px;">Account:</span>
                                    <el-input class="account" v-model="loginUser.account" size="large"
                                        placeholder="User name/Email number">
                                    </el-input>
                                </el-form-item>
                                <el-form-item class="login_input_password" prop="password">
                                    <span style="font-size: 20px;margin-top: 50px;">Password:</span>
                                    <el-input class="pwd" placeholder="Password" v-model="loginUser.password" size="large"
                                        show-password autocomplete="off">
                                    </el-input>
                                </el-form-item>
                            </el-form>
                        </div>
                    </transition>
                    <transition name="animate__animated animate__bounce" enter-active-class="animate__fadeInUp"
                        leave-active-class="animate__zoomOut" appear>
                        <!-- 登录注册按钮 -->
                        <div v-show="isShow" class="btnArea">
                            <el-button class="login_btn" type="success" round style="letter-spacing: 5px"
                                :loading="isloading" @click="UserLogin(loginUserRef)">Login</el-button>
                        </div>
                    </transition>
                </div>
                <!-- 注册表单 -->
                <div class="registArea">
                    <transition name="animate__animated animate__bounce" enter-active-class="animate__fadeInUp"
                        leave-active-class="animate__zoomOut" appear>
                        <!--  注册表头-->
                        <div v-show="!isShow" class="rigestTitle">
                            Registration
                        </div>
                    </transition>
                    <transition name="animate__animated animate__bounce" enter-active-class="animate__fadeInUp"
                        leave-active-class="animate__zoomOut" appear>
                        <!--            注册表单-->
                        <div v-show="!isShow" class="registForm">
                            <el-form ref="regUserRef" class="register_form" :model="regUser" :rules="reg_rules" status-icon
                                :label-position="labelPosition">
                                <el-form-item class="regist_input_regUsername" prop="regUsername" style="margin-top: 20px;">
                                    <span style="font-size: 20px;">Username:</span>
                                    <el-input placeholder="Please enter your user name" v-model="regUser.regUsername" size="large"
                                        style="width: 225px;margin-left: 10px" clearable>
                                    </el-input>
                                </el-form-item>
                                <el-form-item class="regist_input_regPwd" prop="regPwd">
                                    <span
                                        style="font-size: 20px;">Password:
                                    </span>
                                    <el-input placeholder="Please enter your password" style="width: 225px;margin-left: 10px"
                                        v-model="regUser.regPwd" show-password size="large" autocomplete="off">
                                    </el-input>
                                </el-form-item>
                                <el-form-item class="regist_input_regRePwd" prop="regRePwd">
                                    <span style="font-size: 20px;">
                                        Confirm password:
                                    </span>
                                    <el-input placeholder="Please enter the password again" style="width: 225px;margin-left: 10px"
                                        v-model="regUser.regRePwd" show-password size="large" autocomplete="off">
                                    </el-input>
                                </el-form-item>
                                <el-form-item class="regist_input_email" prop="email">
                                    <span
                                        style="font-size: 20px;">Email:
                                    </span>
                                    <el-input placeholder="Please enter your email" v-model="regUser.email"
                                        style="width: 225px;margin-left: 8px" clearable size="large">
                                    </el-input>
                                </el-form-item>
                                <el-form-item class="regist_input_email_code" prop="email_code">
                                    <span style="font-size: 20px;">
                                        Code:
                                    </span>
                                    <el-input placeholder="Please enter the email verification code" v-model="regUser.email_code"
                                        style="width: 225px;margin-left: 10px" clearable size="large">
                                        <template #append>
                                            <el-button :loading="loading" :disabled="getCodeBtnDisable" class="get_code"
                                                type="info" size="large" plain @click.prevent="getCode">
                                                {{ codeBtnWord }}
                                            </el-button>
                                        </template>
                                    </el-input>
                                </el-form-item>
                            </el-form>
                        </div>
                    </transition>
                    <transition name="animate__animated animate__bounce" enter-active-class="animate__fadeInUp"
                        leave-active-class="animate__zoomOut" appear>
                        <!--            注册按钮-->
                        <div v-show="!isShow" class="registBtn">
                            <el-button type="success" round style="letter-spacing: 5px"
                                @click="userRegister(regUserRef)">Register</el-button>
                        </div>
                    </transition>
                </div>
                <!-- 信息展示界面 -->
                <div id="aaa" class="showInfo" :style="{
                    borderTopRightRadius: styleObj.bordertoprightradius,
                    borderBottomRightRadius: styleObj.borderbottomrightradius,
                    borderTopLeftRadius: styleObj.bordertopleftradius,
                    borderBottomLeftRadius: styleObj.borderbottomleftradius,
                    right: styleObj.rightDis
                }" ref="showInfoView">

                    <transition name="animate__animated animate__bounce" enter-active-class="animate__fadeInUp"
                        leave-active-class="animate__zoomOut" appear>
                        <!-- 没有用户输入用户名或者找不到用户名的时候 -->
                        <div v-show="isShow"
                            style="display: flex;flex-direction: column;align-items: center;justify-content: center;width: 100%;height: 100%">
                            <!-- 欢迎语 -->
                            <div class="info_title">
                                Welcome to register and log in to the system
                            </div>
                            <!-- 欢迎图片 -->
                            <div style="flex: 2;">
                                <el-button class="info_button" type="success" round
                                    @click="changeToRegiest">Don't have an account yet? <br /> Click to register</el-button>
                            </div>
                        </div>
                    </transition>
                    <!-- 用户输入用户名时展示头像以及姓名 -->
                    <!--           <div>-->

                    <!--           </div>-->
                    <transition name="animate__animated animate__bounce" enter-active-class="animate__fadeInUp"
                        leave-active-class="animate__zoomOut" appear>
                        <!-- 用户注册的时候展示信息 -->
                        <div v-show="!isShow"
                            style="display: flex;flex-direction: column;align-items: center;justify-content: center;width: 100%;height: 100%">
                            <!-- 欢迎语 -->
                            <div class="info_title">
                                Welcome to register
                            </div>
                            <!-- 欢迎图片 -->
                            <div style="flex: 2">
                                <el-button class="info_button" type="success" round
                                    @click="changeToLogin">Existing account? Click to log in</el-button>
                            </div>
                        </div>
                    </transition>
                </div>
            </div>

        </div>
    </body>
</template>
  
<script setup lang="ts">
import 'animate.css';
// eslint-disable-next-line no-unused-vars
// import { Axios as request } from "axios";
import type { FormProps } from 'element-plus'
import { useCookies } from 'vue3-cookies'
import { getcode, login, register } from '@/api/login'
import { getInfo } from '@/api/user'
import type { FormInstance, FormRules } from 'element-plus'
import { useTokenStore } from '@/store/modules/mytoken'
import { useRouter, useRoute } from 'vue-router'
import useUserInfoStore from '@/store/modules/userInfo'


const router = useRouter()
const route = useRoute()
const store = useTokenStore()
const { cookies } = useCookies()
//看看用不用转成用户对象
const labelPosition = ref<FormProps['labelPosition']>('left')


const getCodeBtnDisable = ref(false)
const loading = ref(false)
const isloading = ref(false)
const waitTime = ref(61)
const codeBtnWord = ref('Get code')
const loginUserRef = ref<FormInstance>()
const regUserRef = ref<FormInstance>()
const userInfoStore = useUserInfoStore()


interface loginUserRule {
    account: string,
    password: string,
}

const loginUser = reactive<loginUserRule>({
    account: "",
    password: "",
})

const login_rules = reactive<FormRules<loginUserRule>>({
    account: [
        { required: true, message: 'Please input the User name', trigger: 'blur' },
        { min: 2, max: 25, message: 'Length should be 3 to 25', trigger: 'blur' },
    ],
    password: [
        { required: true, message: 'Please input the password', trigger: 'blur' },
        { min: 6, max: 16, message: 'Length should be 6 to 16', trigger: 'change' },
    ],
})

interface regUserRule {
    regUsername: string,
    regPwd: string,
    regRePwd: string,
    email: string,
    email_code: string,
}

const regUser = reactive<regUserRule>({
    regUsername: "",
    regPwd: "",
    regRePwd: "",
    email: "",
    email_code: "",
})

const validatePass = (rule: any, value: any, callback: any) => {
    if (value === '') {
        callback(new Error('Please input the password'))
    } else {
        if (regUser.regRePwd !== '') {
            if (!regUserRef.value) return
            regUserRef.value.validateField('regRePwd', () => null)
        }
        callback()
    }
}
const validatePass2 = (rule: any, value: any, callback: any) => {
    if (value === '') {
        callback(new Error('Please input the password again'))
    } else if (value !== regUser.regPwd) {
        callback(new Error("Two inputs don't match!"))
    } else {
        callback()
    }
}

const validateEmail_code = (rule: any, value: any, callback: any) => {
    if (value.length !== 4) {
        callback(new Error('Please input Please enter the 4 digit verification code'))
    } else {
        callback()
    }
}

const reg_rules = reactive<FormRules<regUserRule>>({
    regUsername: [
        { required: true, message: 'Please input the User name', trigger: 'blur' },
        { min: 2, max: 25, message: 'Length should be 3 to 25', trigger: 'blur' },
    ],
    regPwd: [
        {
            validator: validatePass, required: true, trigger: 'blur'
        },
        { min: 6, max: 16, message: 'Length should be 6 to 16', trigger: 'blur' },
    ],
    regRePwd: [
        {
            validator: validatePass2, required: true, trigger: 'blur'
        },
        { min: 6, max: 16, message: 'Length should be 6 to 16', trigger: 'blur' },
    ],
    email: [
        {
            type: "email", message: "Email is incorrect", required: true, trigger: "blur",
        }
    ],
    email_code: [
        {
            required: true, message: 'Please input the email_code', trigger: 'blur',
        },
        {
            validator: validateEmail_code, trigger: 'blur',
        }
    ],
})

const styleObj = ref({
    bordertoprightradius: '15px',
    borderbottomrightradius: '15px',
    bordertopleftradius: '0px',
    borderbottomleftradius: '0px',
    rightDis: '0px'
})
const isShow = ref(true)

onMounted(() => {
})

const changeToRegiest = () => {
    styleObj.value.bordertoprightradius = '0px'
    styleObj.value.borderbottomrightradius = '0px'
    styleObj.value.bordertopleftradius = '15px'
    styleObj.value.borderbottomleftradius = '15px'
    styleObj.value.rightDis = '50%'
    isShow.value = !isShow.value
}

const changeToLogin = () => {
    styleObj.value.bordertoprightradius = '15px'
    styleObj.value.borderbottomrightradius = '15px'
    styleObj.value.bordertopleftradius = '0px'
    styleObj.value.borderbottomleftradius = '0px'
    styleObj.value.rightDis = '0px'
    isShow.value = !isShow.value
}

//用户登录
const UserLogin = async (formEl: FormInstance | undefined) => {
    isloading.value = true
    if (!formEl) return
    let Form = new FormData();
    Form.append("account", loginUser.account);
    Form.append("password", loginUser.password);

    await formEl.validate((valid, fields) => {
        if (valid) {
            login(Form).then(res => {
                if (res.data.code === 200) {
                    ElMessage.success("Login successful!")
                    store.saveToken(res.data.data)
                    getInfo().then(res => {
                        if (res.data.code === 200) {
                            userInfoStore.setUserInfo({logined: true, uid: res.data.data.id, username: res.data.data.username})
                            isloading.value = false
                            router.push((route.query.redirect as string) || '/')
                        } else {
                            ElMessage.error("Failed to obtain user information!")
                            router.push('/login/')
                        }
                    })
                } else if (res.data.code === 300) {
                    isloading.value = false
                    ElMessage.error("Account or password error!")
                } else if (res.data.code === 500) {
                    if (res.data.error_message['account']) {
                        isloading.value = false
                        ElMessage.error(res.data.error_message['account'][0])
                    }
                    else if (res.data.error_message['password']) {
                        isloading.value = false
                        ElMessage.error(res.data.error_message['password'][0])
                    }
                    else {
                        isloading.value = false
                        ElMessage.error("Login failed!")
                    }
                }
            })
        } else {
            console.log('error submit!!', fields)
            isloading.value = false
            return
        }
    })
}

//倒计时方法
const emailTimer = () => {
    const millisecond = new Date().getTime()
    waitTime.value--
    getCodeBtnDisable.value = true
    codeBtnWord.value = `${waitTime.value}s`
    const timer = setInterval(function () {
        if (waitTime.value > 1 && waitTime.value <= 60) {
            waitTime.value--
            codeBtnWord.value = `${waitTime.value}s`
            const expiresTime = new Date(millisecond + waitTime.value * 1000)
            cookies.set(regUser.email, waitTime.value.toString(), { path: '/', expires: expiresTime })
        } else {
            cookies.remove(regUser.email)
            clearInterval(timer)
            codeBtnWord.value = 'Get code'
            getCodeBtnDisable.value = false
            waitTime.value = 61
        }
    }, 1000)
}

// 获取验证码
const getCode = () => {
    const timewait = cookies.get(regUser.email);
    if (timewait !== undefined && parseInt(timewait) > 0) {
        waitTime.value = parseInt(timewait);
        emailTimer();
        return ElMessage.info("Please do not obtain the verification code again after waiting!")
    } else {
        cookies.remove(regUser.email)
        getcode(regUser.email).then(res => {
            if (res.data.code === 200) {
                emailTimer();
            } else {
                ElMessage.error("Please do not obtain the verification code again after waiting!")
            }
        })
        emailTimer();
    }

    // console.log('获取cookie的值：' + this.$cookies.get(this.phone.phone))
}


//用户注册
const userRegister = async (formEl: FormInstance | undefined) => {
    if (!formEl) return
    let Form = new FormData();
    Form.append("username", regUser.regUsername);
    Form.append("password", regUser.regPwd);
    Form.append("password_confirm", regUser.regRePwd);
    Form.append("email", regUser.email);
    Form.append("captcha", regUser.email_code);

    await formEl.validate((valid, fields) => {
        if (valid) {
            console.log(regUser)
            register(Form).then(res => {
                if (res.data.code === 200) {
                    ElMessage.success("Registration successful!")
                    waitTime.value = 61
                    cookies.remove(regUser.email)
                    regUser.regUsername = ''
                    regUser.regRePwd = ''
                    regUser.regPwd = ''
                    regUser.email = ''
                    regUser.email_code = ''
                    changeToLogin()
                } else if (res.data.code === 400) {
                    ElMessage.error("Registration failed!")
                } else if (res.data.code === 500) {
                    console.log(res.data.error_message[0])
                    if (res.data.error_message['captcha']) {
                        ElMessage.error("Verification code error!")
                    } else if (res.data.error_message['username']) {
                        ElMessage.error('Username already exists!')
                    } else if (res.data.error_message['email']) {
                        ElMessage.error('The email already exists!')
                    } else {
                        ElMessage.error("Registration failed!")
                    }
                }
            })
        } else {
            console.log('error submit!!', fields)
            return
        }
    })
}

</script>
  
<style lang="scss" scoped>
body {
    height: 100%;
    background-image: url("@/assets/background/bg.png");
    background-attachment: fixed;
    background-position: center;
    background-size: cover;
    background-repeat: no-repeat;
}

.base {
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;

}

.loginAndRegist {
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
}

.loginArea {
    background-color: rgba(255, 255, 255, 0.8);
    border-top-left-radius: 15px;
    border-bottom-left-radius: 15px;
    height: 640px;
    width: 560px;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    overflow: hidden;
}

.registArea {
    border-top-right-radius: 15px;
    border-bottom-right-radius: 15px;
    height: 640px;
    width: 560px;
    background-color: rgba(255, 255, 255, 0.8);
    z-index: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    // align-items: center;
}

.showInfo {
    border-top-right-radius: 15px;
    border-bottom-right-radius: 15px;
    position: absolute;
    height: 640px;
    width: 560px;
    z-index: 2;
    top: 0;
    right: 0;
    background-image: url("@/assets/background/fg.png");
    background-size: 90%;
}

.showInfo:hover {
    background-size: 100%;
    background-position: -50px -50px;
}

.title {
    width: 70%;
    flex: 1;
    border-bottom: 1px solid #257B5E;
    display: flex;
    align-items: center;
    color: #257B5E;
    font-weight: bold;
    font-size: 35px;
    display: flex;
    justify-content: center;
}

#aaa {
    transition: 0.3s linear;
}

.pwdArea {
    width: 100%;
    flex: 1;
    display: flex;
    flex-direction: column;
    font: 1.3em sans-serif;
    color: #257B5E;
    margin-top: 50px;
}

.pwdArea input {
    outline: none;
    height: 10%;
    border-radius: 13px;
    padding-left: 10px;
    font-size: 20px;
    border: 1px solid gray;
}

.pwdArea input:focus {
    border: 2px solid #257B5E;
}

.login_form {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;;
}

.login_input {
    display: flex;
    width: 100%;
    justify-content: center;
    align-items: center;
    margin-left: 130px;
}

.login_input .account {
    width: 40%;
    height: 35%;
    margin-left: 10px;
    font-size: 16px;
}

.login_input_password {
    display: flex;
    width: 100%;
    justify-content: center;
    align-items: center;
    margin-left: 120px;
}

.login_input_password .pwd {
    width: 40%;
    height: 35%;
    margin-top: 50px;
    margin-left: 10px;
    font-size: 16px;
}

.btnArea {
    flex: 1;
    width: 100%;
    display: flex;
    justify-content: space-around;
    align-items: center;
}

.btnArea .login_btn {
    background-color: #0367a6;
    background-image: linear-gradient(90deg, #0367a6 0%, #008997 74%);
    border-radius: 20px;
    border: 1px solid 0367a6;
    color: #e9e9e9;
    cursor: pointer;
    font-size: 1.5rem;
    font-weight: bold;
    letter-spacing: 0.1rem;
    padding: 1.2rem 3rem;
    text-transform: uppercase;
    transition: transform 80ms ease-in;
}

.rigestTitle {
    width: 100%;
    flex: 1;
    color: #257B5E;
    font-weight: bold;
    font-size: 35px;
    display: flex;
    justify-content: center;
    align-items: center;
    border-bottom: 1px solid #257B5E;
}

.registForm {
    flex: 2;
    display: flex;
    flex-direction: column;
    color: #257B5E;
    font-size: 20px;
}

.registForm input {
    outline: none;
    height: 30%;
    border-radius: 13px;
    padding-left: 10px;
    font-size: 16px;
    border: 1px solid gray;
    display: flex;
    justify-content: center;
    align-items: center;
}

.registForm input:focus {
    border: 2px solid #257B5E;
}

.register_form {
    width: 100%;
    height: 100%;
}

.regist_input {
    display: flex;
    width: 100%;
    justify-content: center;
    align-items: center;
}

.regist_input_regUsername{
    width: 100%;
    margin-left: 130px;
}

.regist_input_regPwd{
    width: 100%;
    margin-left: 136px;
}

.regist_input_regRePwd{
    width: 100%;
    margin-left: 52px;
}

.regist_input_email{
    width: 100%;
    margin-left: 175px;
}

.regist_input_email_code{
    width: 100%;
    margin-left: 175px;
}

.registBtn {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}

.registBtn .el-button {
    background-color: #0367a6;
    background-image: linear-gradient(90deg, #0367a6 0%, #008997 74%);
    border-radius: 20px;
    border: 1px solid 0367a6;
    color: #e9e9e9;
    cursor: pointer;
    font-size: 1.5rem;
    font-weight: bold;
    letter-spacing: 0.1rem;
    padding: 1.2rem 3rem;
    text-transform: uppercase;
    transition: transform 80ms ease-in;
}

.get_code {
    padding: 0px 5px;
    min-width: 50px
}

.info_title {
    width: 90%;
    display: flex;
    flex: 2;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: 45px;
    color: #FFFFFF;
    font-weight: bold

}

.info_button {
    width: 90%;
    height: 18%;
    background-color: #257B5E;
    border: 1px solid #ffffff;
    font-size: 1.5rem;
    font-weight: bold;
    letter-spacing: 0.1rem;
    padding: 1.2rem 3rem;
}
</style>