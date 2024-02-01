<template>
    <div class="segmentation">
        <div class="step_show">
            <!-- <el-steps :active="step_status" process-status="process" finish-status="success">
                <el-step title="Step 1" :icon="Edit" description="文件上传"/>
                <el-step title="Step 2" :icon="Upload" description="模型推理"/>
                <el-step title="Step 3" :icon="Picture" description="分割结果"/>
            </el-steps> -->

            <n-space vertical>
                <n-steps :current="currentNumber" :status="currentStatus">
                    <template #finish-icon>
                        <n-icon>
                            <md-happy />
                        </n-icon>
                    </template>
                    <template #error-icon>
                        <n-icon>
                            <md-sad />
                        </n-icon>
                    </template>
                    <n-step title="Upload" description="Uploading MRI images for segmentation of brain tumors" />
                    <n-step title="Model-based reasoning"
                        description="Segmentation and labeling of components in brain tumors in MRI images" />
                    <n-step title="Segmentation results" description="Display segmentation results">
                        <template #icon>
                            <n-icon>
                                <md-cafe />
                            </n-icon>
                        </template>
                    </n-step>
                </n-steps>
            </n-space>
        </div>
        <div class="segmentation_createTask">
            <el-button class="btn" type="primary" size="large" @click="test1">Create task</el-button>
        </div>
        <div class="segmentation_createTask_main" v-show="task_status.status === 'uploading'">
            <div class="task">
                <el-row class="task_name">
                    <el-input v-model="task_name" class="w-50 m-2" placeholder="Please input the task name" size="large"
                        :focus="check_input1">
                        <template #prepend>Task name:</template>
                    </el-input>
                    <span class="task_info" v-if="!info_show">Task name cannot be empty!</span>
                </el-row>
                <el-upload class="upload-demo" ref="uploadFiles" multiple drag accept=".nii" action=""
                    v-model:file-list="files" :auto-upload="false" :on-change="handleChange"
                    :before-upload="beforeAvatarUpload" :limit="4" :on-exceed="handleExceed">
                    <el-icon class="el-icon--upload">
                        <IEpupload-filled />
                    </el-icon>
                    <div class="el-upload__text">
                        Drop file here or <em>click to upload</em>
                    </div>
                    <template #tip>
                        <div class="el-upload__tip">
                            Please upload an NII format file containing four sequences: Flair, T1, T2, and T1ce
                        </div>
                    </template>
                </el-upload>
            </div>
            <el-button class="create" type="success" size="large" @click="submitUpload">
                Create Task
            </el-button>
            <!-- <el-card>
                <el-button @click="request">发起请求</el-button>
            </el-card> -->
        </div>
        <div class="segmentation_modeling_main" v-loading="fullscreenLoading"
            element-loading-text="Inferring model in progress, please wait..." :element-loading-spinner="svg"
            element-loading-background="rgba(255,251,231, 0.7)" element-loading-svg-view-box="-10, -10, 50, 50"
            v-show="task_status.status === 'modeling'">
        </div>
        <div class="segmentation_result_main" v-show="task_status.status === 'resulting'">
            <div class="segmentation_result_content">
                <!-- <div class="segmentation_result_content_left">
                    <div class="segmentation_result_content_left_title">
                        <span>Original image</span>
                    </div>
                    <div class="radios">
                        <span>Sequence</span>
                        <el-radio-group v-model="radio1" @change="radio_change">
                            <el-radio label="flair" border>flair</el-radio>
                            <el-radio label="t1" border>t1</el-radio>
                            <el-radio label="t2" border>t2</el-radio>
                            <el-radio label="t1ce" border>t1ce</el-radio>
                        </el-radio-group>
                    </div>
                    <div class="Luminance-slider-block">
                        <span>Luminance</span>
                        <el-slider v-model="ori_luminance" @input="drawLeftImage(radio1)" :min="-100" />
                    </div>
                    <div class="segmentation_result_content_left_img">
                        <canvas id="left_img" class="segmentation_result_left_img"></canvas>
                    </div>
                    <div class="slider-block">
                        <span>Serial Number</span>
                        <el-slider show-input v-model="original_number" @input="drawLeftImage(radio1)"
                            :max="ori_pic_num_max" />
                    </div>
                </div> -->
                <div class="segmentation_result_content_left">
                    <div class="segmentation_result_content_left_title">
                        <span>Segmentation results</span>
                    </div>
                    <div class="radios">
                        <span>Sequence</span>
                        <el-radio-group v-model="radio2" @change="radio_change">
                            <el-radio label="flair" border>flair</el-radio>
                            <el-radio label="t1" border>t1</el-radio>
                            <el-radio label="t2" border>t2</el-radio>
                            <el-radio label="t1ce" border>t1ce</el-radio>
                        </el-radio-group>
                    </div>
                    <div class="Luminance-slider-block">
                        <span>Luminance</span>
                        <el-slider v-model="seg_luminance" @input="drawLeftImage(radio2)" :min="-100" />
                    </div>
                    <div class="segmentation_result_content_left_img">
                        <canvas id="left_bottom_images" class="segmentation_result_left_bottom_img"></canvas>
                        <canvas id="left_img" class="segmentation_result_left_img"
                            :style="{ opacity: transparency }"></canvas>
                        <div class="vertical_slider-block">
                            <el-slider v-model="transparency" @input="drawLeftImage(radio2)" :format-tooltip="formatTooltip"
                                vertical />
                            <span>Transparency</span>
                        </div>
                    </div>
                    <div class="slider-block">
                        <span>Serial Number</span>
                        <el-slider v-model="seg_number" :max="seg_number_max" @input="drawLeftImage(radio2)" show-input />
                    </div>
                </div>
                <div class="result_content_right">
                    <div class="classification_result_info">
                        <el-card class="classification_result" :body-style="{ height: '50%' }">
                            <template #header>
                                <div class="info_icon">
                                    <svg-icon name="结果" width="150px" height="150px"></svg-icon>
                                    <span>Classification Results</span>
                                </div>
                            </template>
                            <div class="text item">
                                <span>IDH Wild type:</span>
                                <span>{{ classification_result['idh_wild'] }}</span>
                            </div>
                            <div class="text item">
                                <span>IDH Mutant type:</span>
                                <span>{{ classification_result['idh_mutant'] }}</span>
                            </div>
                            <div class="text item">
                                <span>1p/19q intac:</span>
                                <span>{{ classification_result['1p/19q intac'] }}</span>
                            </div>
                            <div class="text item">
                                <span>1p/19q codel:</span>
                                <span>{{ classification_result['1p/19q codel'] }}</span>
                            </div>
                            <div class="text item">
                                <span>GBM(high grade glioma):</span>
                                <span>{{ classification_result['grade_HGG'] }}</span>
                            </div>
                            <div class="text item">
                                <span>LGG(low grade glioma):</span>
                                <span>{{ classification_result['grade_LGG'] }}</span>
                            </div>
                            <!-- <div class="text item">
                                <span>Confidence level:</span>
                                <span>{{}}</span>
                            </div> -->
                            <div class="info_button">
                                <el-button size="large" type="primary" @click="table = true">AI Consulting</el-button>
                            </div>

                        </el-card>
                    </div>

                    <div class="segmentation_result_info">
                        <div class="segmentation_result">
                            <div class="info_icon">
                                <svg-icon name="结果对比分析" width="150px" height="150px"></svg-icon>
                                <div claas="info_info">
                                    <span class="info_title">Segmentation Results<br></span>
                                    <span class="info_p">(Proportion of each subregion of tumors)</span>
                                </div>
                            </div>
                            <div class="divider"></div>
                            <div class="ratio_info">
                                <div class="text">
                                    <span>NCR Ratio(Necrotic area):</span>
                                    <span>{{ NCR_ratio }}</span>
                                </div>
                                <div class="text">
                                    <span>ED Ratio(Edema area):</span>
                                    <span>{{ ED_ratio }}</span>
                                </div>
                                <div class="text">
                                    <span>ET Ratio(Enhancing tumor area):</span>
                                    <span>{{ ET_ratio }}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <el-drawer v-model="table" title="I have a nested table inside!" direction="rtl" size="50%">
                    <el-table :data="gridData">
                        <el-table-column property="date" label="Date" width="150" />
                        <el-table-column property="name" label="Name" width="200" />
                        <el-table-column property="address" label="Address" />
                    </el-table>
                </el-drawer>
            </div>
        </div>
    </div>
</template>

<script setup lang='ts'>
import {
    MdArrowRoundBack,
    MdArrowRoundForward,
    MdCafe,
    MdHappy,
    MdSad
} from '@vicons/ionicons4'
import { StepsProps } from 'naive-ui'
import { genFileId } from 'element-plus'
import type { UploadInstance, UploadProps, UploadUserFile, UploadRawFile } from 'element-plus'
import { onBeforeUnmount } from 'vue'
import { upload, test } from '@/api/upload';
import { modeling } from '@/api/modeling'
import { seg_history_search, search_autocomplete } from '@/api/history'
// import { useTokenStore } from '@/store/modules/mytoken';
import useUserInfoStore from '@/store/modules/userInfo';
// import request from '@/utils/request';
import dataURLtoBlob from '@/utils/base64Topng';

// const store = useTokenStore()
const userInfoStore = useUserInfoStore();
const controller = new AbortController()
const task_name = ref<string>('')
const upload_task_name = ref<string>('')
const info_show = ref<boolean>(true)
const uploadFiles = ref<UploadInstance>()
const current = ref<number | null>(1)
const currentStatus = ref<StepsProps['status']>('process')
const task_status = reactive({ status: "uploading" })
const files = ref<UploadUserFile[]>([])
const formData = new FormData()
const fullscreenLoading = ref(true)
const svg = `
        <path class="path" d="
          M 30 15
          L 28 17
          M 25.61 25.61
          A 15 15, 0, 0, 1, 15 30
          A 15 15, 0, 1, 1, 27.99 7.5
          L 15 15
        " style="stroke-width: 4px; fill: rgba(0, 0, 0, 0)"/>
      `
// const radio1 = ref<string>('flair');
const radio2 = ref<string>('flair');
const table = ref(false)
const gridData = [
    {
        date: '2016-05-02',
        name: 'Peter Parker',
        address: 'Queens, New York City',
    },
    {
        date: '2016-05-04',
        name: 'Peter Parker',
        address: 'Queens, New York City',
    },
    {
        date: '2016-05-01',
        name: 'Peter Parker',
        address: 'Queens, New York City',
    },
    {
        date: '2016-05-03',
        name: 'Peter Parker',
        address: 'Queens, New York City',
    },
]

type ori_pic = {
    [key: string]: string[]
}
type ori_pic_num = {
    [key: string]: number
}
type classific_result = {
    [key: string]: number
}
const original_pic = ref<ori_pic>({})
// const original_number = ref<number>(0)
const original_number_max = ref<ori_pic_num>({})
const seg_pic = ref<string[]>([])
const seg_number = ref<number>(0)
const seg_number_max = ref<number>(0)
// const ori_luminance = ref<number>(0)
const seg_luminance = ref<number>(0)
const transparency = ref<number>(50)
const NCR_ratio = ref<number>()
const ED_ratio = ref<number>()
const ET_ratio = ref<number>()
const seg_result = ref<classific_result>({})
const classification_result = ref<classific_result>({})
// var i = 0


const test1 = async () => {
    // console.log(i++)
    await search_autocomplete({ userId: userInfoStore.getuserInfo.uid }).then(res => {
        console.log(res)
    }).catch(err => {
        console.log(err)
    })
}

onBeforeUnmount(() => {
    controller.abort()
})

const radio_change = () => {
    // drawLeftImage(radio1.value)
    drawLeftImage(radio2.value)
}

const currentNumber = computed(() => {
    // 在这里进行类型断言
    return current.value as number;
});

// const ori_pic_num_max = computed(() => {
//     return original_number_max.value[radio1.value] as number;
// });

watch(task_status, async (newVal, oldVal) => {
    if (newVal.status === 'modeling') {
        fullscreenLoading.value = true
        console.log(upload_task_name.value)
        try {
            await modeling({ 'task_name': upload_task_name.value, 'signal': controller.signal }).then(res => {
                if (res.data.code === 200) {
                    console.log(res.data)
                    classification_result.value = JSON.parse(res.data.data.classific_result)
                    seg_result.value = JSON.parse(res.data.data.seg_result)
                    NCR_ratio.value = seg_result.value['NCR_ratio']
                    ED_ratio.value = seg_result.value['ED_ratio']
                    ET_ratio.value = seg_result.value['ET_ratio']
                    Object.keys(res.data.data.original_num).forEach(element => {
                        original_pic.value[element] = []
                        if (element === 'flair' || element == 't1' || element == 't2' || element == 't1ce') {
                            original_number_max.value[element] = res.data.data.original_num[element]
                            for (let i = 0; i < original_number_max.value[element]; i++) {
                                original_pic.value[element].push(URL.createObjectURL(dataURLtoBlob(res.data.data.original_images[element][i])))
                            }
                        } else {
                            original_number_max.value[element] = 0
                            ElMessage.error(element + "Sequence does not exist!")
                        }
                    });
                    seg_number_max.value = res.data.data.seg_num
                    for (let i = 0; i < seg_number_max.value; i++) {
                        seg_pic.value.push(URL.createObjectURL(dataURLtoBlob(res.data.data.seg_images[i])))
                    }
                    upload_task_name.value = ''
                    task_status.status = 'resulting'
                    current.value = 3
                    fullscreenLoading.value = false
                    // drawLeftImage(radio1.value)
                    drawLeftImage(radio2.value)
                } else {
                    ElMessage.error("Model inference failed!")
                    upload_task_name.value = ''
                    uploadFiles.value!.clearFiles()
                    task_status.status = 'uploading'
                    fullscreenLoading.value = false
                }
            }).catch(err => {
                console.log(err)
                ElMessage.error("Model inference failed!")
                upload_task_name.value = ''
                uploadFiles.value!.clearFiles()
                task_status.status = 'uploading'
                fullscreenLoading.value = false
            })
        } catch (error) {
            console.log(error)
            ElMessage.error("Model inference failed!")
            upload_task_name.value = ''
            uploadFiles.value!.clearFiles()
            task_status.status = 'uploading'
            fullscreenLoading.value = false
        }
    }
    console.log(newVal, oldVal)
})

const handleExceed: UploadProps['onExceed'] = (files) => {
    uploadFiles.value!.clearFiles()
    const file = files[0] as UploadRawFile
    file.uid = genFileId()
    uploadFiles.value!.handleStart(file)
    formData.delete('files')
}

const beforeAvatarUpload = (files: any) => {
    // 检测文件类型和大小
    for (let i = 0; i < files.length; i++) {
        let isFileType = files[i].name.split(".")[files[i].name.split(".").length - 1] === "nii";
        let isFileSize = files[i].size / 1024 / 1024 < 25;

        if (!isFileType) ElMessage.error(files[i].name + "The format does not meet the requirements, the uploaded file can only be a NII file!");
        if (!isFileSize) ElMessage.error(files[i].name + "The file size does not meet the requirements. The uploaded file size cannot exceed 25MB!");

        if (!isFileType || !isFileSize) {
            return false;
            break;
        }
    }

    return true;
}

const check_input1 = () => {
    info_show.value = true;
}

const submitUpload = async () => {
    if (files.value.length === 0) {
        ElMessage.error("请上传文件!");
        return;
    } else {
        if (task_name.value === "") {
            info_show.value = false;
            ElMessage.error("Task name cannot be empty!");
            return;
        } else if (beforeAvatarUpload(files.value)) {
            files.value.forEach((file) => {
                if (file.raw) {
                    formData.append("files", file.raw);
                } else {
                    return ElMessage.error("File upload failed!");
                }
            });
            formData.append('task_name', task_name.value)
            formData.append('upload_time', new Date().toLocaleString())
            await upload(formData).then(res => {
                if (res.data.code === 200) {
                    ElMessage.success("Task created successfully!")
                    current.value = 2
                    currentStatus.value = 'process'
                    task_status.status = 'modeling'
                    uploadFiles.value!.clearFiles()
                    upload_task_name.value = task_name.value
                    task_name.value = ''
                    formData.delete('files')
                    formData.delete('task_name')
                } else {
                    console.log(res.data)
                    formData.delete('files')
                    formData.delete('task_name')
                    formData.delete('upload_time')
                    ElMessage.error(res.data.error_message + ",Task creation failed!")
                }
            }).catch(err => {
                console.log(err)
            })
        }
    }
}


const handleChange = (file: any, files: any) => {
    let existFile = files.slice(0, files.length - 1).find((f: any) => f.name === file.name);
    if (existFile) {
        ElMessage.error('The current file already exists!');
        files.pop();
    }
}

// const add_luminance = () => {
//     luminance.value += 10
//     change_light(radio1.value, radio2.value)
// }

// const reduce_luminance = () => {
//     luminance.value -= 10
//     change_light(radio1.value, radio2.value)
// }

const formatTooltip = (val: number) => {
    return val / 100
}

// const drawLeftImage = (sequence: string) => {
//     const canvas = document.getElementById("left_img") as HTMLCanvasElement;
//     const ctx = canvas!.getContext("2d");
//     const img = new Image();
//     img.src = original_pic.value[sequence][original_number.value];
//     img.onload = () => {
//         canvas!.width = img.width;//设置画布大小和图片相同，否则画布有默认大小
//         canvas!.height = img.height;

//         ctx!.drawImage(img, 0, 0, img.width, img.height);//画图片
//         const imgData = ctx!.getImageData(0, 0, img.width, img.height);//获取图片数据
//         const data = imgData.data;//获取图片数据的data部分

//         for (var i = 0; i < data.length; i += 4) {
//             data[i + 0] += ori_luminance.value // r，红通道
//             data[i + 1] += ori_luminance.value // g，绿通道
//             data[i + 2] += ori_luminance.value // b，蓝通道
//         }
//         ctx!.clearRect(0, 0, canvas!.width, canvas!.width);//清除画布
//         ctx!.putImageData(imgData, 0, 0);//重新绘制图片数据
//     }
// }

const left_bottom_images = (sequence: string) => {
    const canvas = document.getElementById("left_bottom_images") as HTMLCanvasElement;
    const ctx = canvas!.getContext("2d");
    const img = new Image();

    img.src = original_pic.value[sequence][seg_number.value];
    img.onload = () => {
        canvas!.width = img.width;//设置画布大小和图片相同，否则画布有默认大小
        canvas!.height = img.height;
        ctx!.drawImage(img, 0, 0, img.width, img.height);//画图片
        const imgData = ctx!.getImageData(0, 0, img.width, img.height);//获取图片数据
        const data = imgData.data;//获取图片数据的data部分

        for (var i = 0; i < data.length; i += 4) {
            data[i + 0] += seg_luminance.value // r，红通道
            data[i + 1] += seg_luminance.value // g，绿通道
            data[i + 2] += seg_luminance.value // b，蓝通道
        }
        ctx!.clearRect(0, 0, canvas!.width, canvas!.width);//清除画布
        ctx!.putImageData(imgData, 0, 0);//重新绘制图片数据
    }
}

const drawLeftImage = (sequence: string) => {
    const canvas = document.getElementById("left_img") as HTMLCanvasElement;
    const ctx = canvas!.getContext("2d");
    const img = new Image();

    img.src = seg_pic.value[seg_number.value];
    img.onload = () => {
        canvas!.width = img.width;//设置画布大小和图片相同，否则画布有默认大小
        canvas!.height = img.height;
        ctx!.drawImage(img, 0, 0, img.width, img.height);//画图片

        const img2Data = ctx!.getImageData(0, 0, img.width, img.height);//获取图片数据
        const data = img2Data.data;//获取图片数据的data部分
        for (let i = 0; i < data.length; i += 4) {
            let red = data[i];
            let green = data[i + 1];
            let blue = data[i + 2];
            let alpha = data[i + 3];

            if (data[i] === 0 && data[i + 1] === 0 && data[i + 2] === 0) {
                data[i + 3] = 0;
            } else {
                data[i] = red * transparency.value / 100;
                data[i + 1] = green * transparency.value / 100;
                data[i + 2] = blue * transparency.value / 100;
                data[i + 3] = alpha * transparency.value / 100;
            }
        }
        ctx!.clearRect(0, 0, canvas!.width, canvas!.width);//清除画布
        ctx!.putImageData(img2Data, 0, 0);//重新绘制图片数据
    }

    left_bottom_images(sequence)
}

// const show_request = async () => {
//     await request({
//         url: '/dashboard/show_images/',
//         method: 'get',
//         headers: {
//             'Authorization': 'Bearer ' + store.token.access_token,
//         }
//     }).then(res => {
//         original_number_max.value = res.data.data.original_num
//         seg_number_max.value = res.data.data.seg_num
//         for (let i = 0; i < original_number_max.value; i++) {
//             original_pic.value.push(URL.createObjectURL(dataURLtoBlob(res.data.data.original_images[i])))
//             seg_pic.value.push(URL.createObjectURL(dataURLtoBlob(res.data.data.seg_images[i])))
//         }
//         drawLeftImage()
//         drawRightImage()
//     }).catch(err => {
//         console.log(err)
//     })
// }

</script>

<style lang="scss" scoped>
body {
    overflow: hidden;
}

.segmentation {
    width: 100%;
    height: auto;
    // min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.step_show {
    margin-top: 10px;
    margin-left: auto;
    width: 80%;
}

.step_show .n-space {
    width: 100%;
}

.segmentation_createTask_main {
    margin-top: 100px;
    width: 60%;
    height: 100%;
    display: flex;
    flex-direction: column;
    position: relative;
}

.task {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.task .task_name {
    width: 50%;
    margin-left: 130px;
    margin-bottom: 20px;
}

.task .task_name .task_info {
    margin-left: 135px;
    color: red;
}

.upload-demo {
    width: 80%;
    height: 50%;
    margin-top: 30px;
    margin: 0 auto;
}

:deep(.el-upload .el-upload-dragger) {
    width: 100%;
    height: 30vh;
}

:deep(.el-upload .el-upload-dragger .el-upload__text) {
    font-size: 20px;
}

.upload-demo .el-upload__tip {
    font-size: 18px;
}

.upload-demo .el-icon--upload {
    font-size: 180px;
    color: #409EFF;
}

.create {
    font-size: 20px;
    width: 150px;
    height: 50px;
    border-radius: 10px;
    background-color: #409EFF;
    color: white;
    border: none;
    margin: 150px auto;
}

.create:hover {
    background-color: #66b1ff;
}

.example-showcase .el-loading-mask {
    z-index: 9;
}

.segmentation_modeling_main {
    margin-top: 50px;
    width: 85%;
    height: 100%;
    // border: #409EFF solid 1px;
}

.segmentation_result_main {
    margin-top: 50px;
    width: 73vw;
    height: 90vh;
    border: #409EFF solid 1px;
}

.segmentation_result_content {
    width: 98%;
    height: 100%;
    display: flex;
    flex-direction: row;
}

// .segmentation_result_content_left {
//     width: 50%;
//     height: 95%;
//     display: flex;
//     flex-direction: column;
// }

// .segmentation_result_content_left_title {
//     width: 100%;
//     height: 10%;
//     display: flex;
//     justify-content: center;
//     align-items: center;
// }

// .segmentation_result_content_left_title span {
//     font-size: 20px;
//     font-weight: bold;
// }

.Luminance-slider-block {
    margin-top: 10px;
    display: flex;
    justify-content: center;
    align-items: center;
    margin-left: 20px;
    width: 100%;
    height: 3%;
}

.Luminance-slider-block span {
    margin-right: 15px;
}

.Luminance-slider-block .el-slider {
    width: 60%;
    margin-top: 0;
    margin-left: 12px;
}

// .segmentation_result_content_left_img {
//     width: 100%;
//     height: 100%;
//     display: flex;
//     justify-content: center;
//     align-items: center;
//     position: relative;
// }

// .segmentation_result_content_left_img .segmentation_result_left_img {
//     position: absolute;
//     margin: auto;
// }

canvas {
    width: 88%;
    height: 94%;
    // background-color: rgb(245, 247, 250);
}

.radios {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-left: 20px;
}

.radios span {
    margin-right: 10px;
}

.segmentation_result_content_left {
    width: 50%;
    height: 95%;
    display: flex;
    flex-direction: column;
}

.segmentation_result_content_left_title {
    width: 100%;
    height: 10%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.segmentation_result_content_left_title span {
    font-size: 20px;
    font-weight: bold;
}

.el-radio-group {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-left: 20px;
}

.segmentation_result_content_left_img {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
}

.segmentation_result_content_left_img .segmentation_result_left_img {
    position: absolute;
    margin: auto;
}

.segmentation_result_content_left_img .segmentation_result_left_bottom_img {
    position: absolute;
    margin: auto;
}

.slider-block {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    height: 5%;
}

.slider-block .el-slider {
    width: 75%;
    margin-top: 0;
    margin-left: 12px;
}

.vertical_slider-block {
    display: flex;
    align-items: center;
    width: 7%;
    height: 100%;
    margin-left: 10px;
    position: absolute;
    right: 3px;
}

.vertical_slider-block span {
    width: 100%;
    writing-mode: vertical-lr;
    transform: rotate(360deg);
}

.vertical_slider-block .el-slider {
    width: 100%;
    height: 75%;
    margin-top: 0;
    margin-left: 10px;
}

.result_content_right {
    width: 50%;
    height: 95%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.result_content_right .classification_result_info {
    width: 100%;
    height: 45%;
    margin-top: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.classification_result_info .classification_result {
    width: 60%;
    height: 100%;
    justify-content: space-around;
}

.classification_result_info .classification_result .info_icon {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    align-items: center;
}

.classification_result_info .classification_result .info_icon svg-icon {
    width: 100%;
    height: 100%;
}

.classification_result_info .classification_result .info_icon span {
    font-size: 30px;
    font-weight: bold;
}

.classification_result_info .classification_result .text {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.classification_result_info .classification_result .text span:nth-child(1) {
    font-size: 20px;
    margin-right: 10px;
}

.classification_result_info .classification_result .text span:nth-child(2) {
    font-size: 20px;
    font-weight: 500;
}

.classification_result_info .classification_result .info_button {
    width: 100%;
    height: 15%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.classification_result_info .classification_result .info_button .el-button {
    width: 50%;
    height: 100%;
    margin-top: 70px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 20px;
    background-color: #409EFF;
    color: white;
    border: none;
}

.classification_result_info .classification_result .el-button:hover {
    background-color: #66b1ff;
}

.result_content_right .segmentation_result_info {
    width: 60%;
    height: 35%;
    margin-top: 50px;
}

.segmentation_result_info .segmentation_result {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    background-color: rgb(147, 252, 243);
    border-radius: 10px;
    box-shadow: 0 0 15px rgba(0, 0, 0, 0.35);
}

.segmentation_result_info .segmentation_result .info_icon {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.segmentation_result_info .segmentation_result .info_icon svg-icon {
    width: 100%;
    height: 100%;
}

.segmentation_result_info .segmentation_result .info_icon .info_info {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}

.segmentation_result_info .segmentation_result .info_icon .info_title {
    font-size: 30px;
    font-weight: bold;
    display: flex;
    text-align: center;
}

.segmentation_result_info .segmentation_result .info_icon .info_p {
    font-size: 18px;
    font-weight: normal;
    display: flex;
    text-align: center;
}

.segmentation_result_info .segmentation_result .divider {
    width: 100%;
    height: 2.5px;
    background-color: rgba(152, 152, 152, 0.324);
    margin-left: 20px;
    margin-right: 20px;
}

.segmentation_result_info .segmentation_result .ratio_info {
    width: 93%;
    height: 100%;
    font-size: 20px;
    // font-weight: bold;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.segmentation_result_info .segmentation_result .ratio_info .text {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
</style>