<template>
    <div class="main">
        <el-row>
            <el-col v-for="(o, index) in history_tasks.length" :key="o" :span="6" :offset="index > 0 ? 1 : 1">
                <el-card :body-style="{ padding: '10px', width: 'auto', height: '88%' }">
                    <!-- <span>{{ o }}</span> -->
                    <el-image :src="get_original_images_path(index)" class="image" fit="fill" />
                    <div class="card_info_header">
                        <el-row :gutter="20">
                            <el-col :span="12">
                                <span>{{ history_tasks[index]['filename'] }}</span>
                            </el-col>
                            <el-col :span="4" :offset="8">
                                <span>{{ history_tasks[index]['type'] }} </span>
                            </el-col>
                        </el-row>
                    </div>
                    <div class="card_info_middle">
                        <time class="time">{{ history_tasks[index]['classification_time'] }}</time>
                        <div class="options">
                            <el-button size="large" dark color="#28ca61" :icon="Search" circle class="show_button"
                                @click="openDialog(history_tasks[index])"></el-button>
                            <el-button size="large" dark color="#d64141" :icon="Delete" circle class="delete_button"
                                @click="delete_task(history_tasks[index])"></el-button>
                        </div>
                    </div>
                    <div class="card_info_footer">
                        <div class="seq_tags">
                            <el-tag v-for="sequence in get_sequence(index)" :key="sequence.label" class="mx-1"
                                :type="sequence.type" effect="light">
                                {{ sequence.label }}
                            </el-tag>
                        </div>
                        <div class="classification_result">
                            <el-row class="category">
                                <el-col :span="8">
                                    <el-text class="idh_wild">{{ "idh_wild: " +
                                        JSON.parse(history_tasks[index]['classification_result'])['idh_wild'] }}</el-text>
                                </el-col>
                                <el-col :span="8">
                                    <el-text class="idh_mutant">{{ "idh_mutant: " +
                                        JSON.parse(history_tasks[index]['classification_result'])['idh_mutant'] }}</el-text>
                                </el-col>
                                <el-col :span="8">
                                    <el-text class="1p/19q intac">{{ "1p/19q intac: " +
                                        JSON.parse(history_tasks[index]['classification_result'])['1p/19q intac'] }}</el-text>
                                </el-col>
                            </el-row>
                            <div class="category">
                                <el-col :span="8">
                                    <el-text class="1p/19q codel">{{ "1p/19q codel: " +
                                        JSON.parse(history_tasks[index]['classification_result'])['1p/19q codel'] }}</el-text>
                                </el-col>
                                <el-col :span="8">
                                    <el-text class="grade_LGG">{{ "LGG:" +
                                        JSON.parse(history_tasks[index]['classification_result'])['grade_LGG'] }}</el-text>
                                </el-col>
                                <el-col :span="8">
                                    <el-text class="grade_HGG">{{ "HGG:" +
                                        JSON.parse(history_tasks[index]['classification_result'])['grade_HGG'] }}</el-text>
                                </el-col>
                            </div>
                        </div>
                    </div>
                </el-card>
            </el-col>
        </el-row>
    </div>
    <div class="footer">
        <el-pagination background layout="prev, pager, next, total" @current-change="handleCurrentChange"
            :page-size="pagination_page_size" :current-page="pagination_page_number" :total="pagination_total" />
    </div>
    <el-dialog claas="ResultModal" v-model="showModalRef" @closed="handleDialogClosed" width="80%" :title="modal_title">
        <el-form class="modal_main" v-loading="fullscreenLoading"
            element-loading-text="Inferring model in progress, please wait..." :element-loading-spinner="svg"
            element-loading-background="rgba(242,250,255, 0.7)" element-loading-svg-view-box="-10, -10, 50, 50">
            <div class="classification_result_main">
                <div class="classification_result_content">
                    <div class="classification_result_content_left">
                        <div class="classification_result_content_left_title">
                            <span>Classification Results</span>
                        </div>
                        <div class="radios">
                            <span>Sequence</span>
                            <el-radio-group v-model="radio1" v-for="sequence in original_sequence_list" :key="sequence"
                                @change="radio_change">
                                <el-radio :label="sequence" :key="sequence" border>{{ sequence }}</el-radio>
                                {{ sequence }}
                            </el-radio-group>
                        </div>
                        <div class="Luminance-slider-block">
                            <span>Luminance</span>
                            <el-slider v-model="seg_luminance" @input="drawLeftImage(radio1)" :min="-100" />
                        </div>
                        <div class="classification_result_content_left_img">
                            <canvas id="left_bottom_images" class="classification_result_left_bottom_img"></canvas>
                            <canvas id="left_img" class="classification_result_left_img"
                                :style="{ opacity: transparency }"></canvas>
                            <div class="vertical_slider-block">
                                <el-slider v-model="transparency" @input="drawLeftImage(radio1)"
                                    :format-tooltip="formatTooltip" vertical />
                                <span>Transparency</span>
                            </div>
                        </div>
                        <div class="slider-block">
                            <span>Serial Number</span>
                            <el-slider v-model="seg_number" :max="ori_pic_num_max" @input="drawLeftImage(radio1)"
                                show-input />
                        </div>
                    </div>
                    <div class="classification_result_divider">
                        <el-divider direction="vertical" />
                    </div>
                    <div class="classification_result_content_right">
                        <div class="result_info">
                            <el-card class="box-card">
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
                                <!-- <el-button size="large" type="primary">AI Consulting</el-button> -->
                            </el-card>
                        </div>
                        <!-- <template #sub-title>
                                <p style="font-size:50px">Result Prediction</p>
                                <p>Result Prediction</p>
                            </template>
                            <template #extra>
                                <el-button size="large" type="primary">Back</el-button>
                            </template>
                        </el-result> -->
                    </div>
                </div>
            </div>
        </el-form>
        <template #footer>
            <span class="dialog-footer">
                <el-button type="primary" size="large" @click="showModalRef = false">
                    Cancel
                </el-button>
            </span>
        </template>
    </el-dialog>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
import { inject } from "vue";
import {
    Delete,
    Search,
} from '@element-plus/icons-vue'
import { classification_history, classification_history_detail, classification_history_delete } from '@/api/history'
import useUserInfoStore from '@/store/modules/userInfo';
import type { TagProps } from 'element-plus'

type Item = { type: TagProps['type']; label: string }

const items = ref<Array<Item>>([
    { type: '', label: 'Flair' },
    { type: 'success', label: 'T1' },
    { type: 'info', label: 'T2' },
    { type: 'danger', label: 'T1ce' },
])

const showModalRef = ref(false);
const userInfoStore = useUserInfoStore();
const history_tasks = ref([]);
const modal_original_pic = ref({
    src: '../../public/',
    pic_number: 0
});
const modal_classified_pic = ref({
    src: '../../public/',
    pic_number: 0
});
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
const modal_title = ref('');
const radio1 = ref<string>('flair');
const pagination_total = ref(0);
const pagination_page_size = ref(6);
const pagination_page_number = ref(1);

type ori_pic = {
    [key: string]: string[]
}
type ori_pic_num = {
    [key: string]: number
}
type classific_result = {
    [key: string]: number
}
const original_sequence_list = ref<string[]>([])
const original_pic = ref<ori_pic>({})
const original_number_max = ref<ori_pic_num>({})
const transparency = ref<number>(50)
const seg_pic = ref<string[]>([])
const seg_number = ref<number>(0)
const seg_number_max = ref<number>(0)
const seg_luminance = ref<number>(0)
const classification_result = ref<classific_result>({})

const reload = inject("reload");

onMounted(async () => {
    await classification_history({ userId: userInfoStore.getuserInfo.uid, page: pagination_page_number.value, page_size: pagination_page_size.value }).then(res => {
        if (res.data.code == 200) {
            console.log(res.data.data)
            history_tasks.value = res.data.data.page_list
            pagination_total.value = res.data.data.total
        } else {
            console.log('获取任务失败')
        }
    })
})

const handleCurrentChange = async (current_page: number) => {
    pagination_page_number.value = current_page
    console.log(pagination_page_number.value)
    await classification_history({ userId: userInfoStore.getuserInfo.uid, page: pagination_page_number.value, page_size: pagination_page_size.value }).then(res => {
        if (res.data.code == 200) {
            console.log(res.data.data)
            history_tasks.value = res.data.data.page_list
            pagination_total.value = res.data.data.total
        } else {
            console.log('获取任务失败')
        }
    })
}

const radio_change = () => {
    drawLeftImage(radio1.value)
}

const ori_pic_num_max = computed(() => {
    return original_number_max.value[radio1.value] as number;
});

const get_sequence = (index: number) => {
    const sequence_str: String[] = []
    const tag_sequence: any[] = []
    const str = history_tasks.value[index]['original_images_sequence'].split(',')
    for (let element of str) {
        sequence_str.push(element.charAt(0).toUpperCase() + element.slice(1))
    }
    items.value.map((item) => {
        if (sequence_str.indexOf(item.label) !== -1) {
            tag_sequence.push(item)
        }
    })
    return tag_sequence
}

const get_original_images_path = (index: number) => {
    return "http://127.0.0.1:5000/" + history_tasks.value[index]['path'] + "/100.png"
}

const delete_task = (task: any) => {
    ElMessageBox({
        title: 'Warning',
        message: h('p', null, [
            h('span', null, 'It will permanently delete the file. Continue?'),
        ]),
        type: 'warning',
        icon: markRaw(Delete),
        showCancelButton: true,
        confirmButtonText: 'OK',
        cancelButtonText: 'Cancel',
        beforeClose: (action, instance, done) => {
            if (action === 'confirm') {
                instance.confirmButtonLoading = true
                instance.confirmButtonText = 'Deleting...'
                setTimeout(async () => {
                    await classification_history_delete({ userId: userInfoStore.getuserInfo.uid, task_name: task.filename })
                        .then((res: any) => {
                            if (res.data.code === 200) {
                                done()
                                setTimeout(() => {
                                    instance.confirmButtonLoading = false
                                }, 100)
                                ElMessage({
                                    type: 'success',
                                    message: 'Delete successfully',
                                })
                                reload()
                            } else {
                                done()
                                setTimeout(() => {
                                    instance.confirmButtonLoading = false
                                }, 100)
                                ElMessage.error('Delete task failed')
                                console.log('Delete task failed')
                            }
                        }).catch((err: any) => {
                            console.log(err)
                        })
                }, 200)
            } else {
                done()
            }
        }
    })
}


const openDialog = async (task: any) => {
    showModalRef.value = true
    fullscreenLoading.value = true
    await classification_history_detail({ userId: userInfoStore.getuserInfo.uid, task_name: task.filename }).then((res: any) => {
        if (res.data.code == 200) {
            console.log(res.data.data)
            fullscreenLoading.value = false
            original_sequence_list.value = res.data.data.original_sequence_list
            original_pic.value = res.data.data.original_images
            original_number_max.value = res.data.data.original_num
            seg_pic.value = res.data.data.seg_images
            seg_number_max.value = res.data.data.seg_num
            modal_title.value = task.filename
            classification_result.value = JSON.parse(task.classification_result)
            drawLeftImage(radio1.value)
        } else {
            console.log('获取任务失败')
        }
    }).catch((err: any) => {
        console.log(err)
    });
}

const handleDialogClosed = () => {
    modal_original_pic.value.src = '/',
        modal_original_pic.value.pic_number = 0,
        modal_classified_pic.value.src = '/',
        modal_classified_pic.value.pic_number = 0
}

const formatTooltip = (val: number) => {
    return val / 100
}


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

</script>

<style lang="scss" scoped>
.main {
    width: 90%;
    height: 98%;
    padding-left: 50px;
}

.footer {
    width: 90%;
    height: 7%;
    padding-left: 50px;
    display: flex;
    justify-content: right;
    align-items: center;
    padding-right: 50px;

}

.modal_main {
    height: 89vh;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.el-card {
    margin-bottom: 10px;
    width: 95%;
    height: 90%;
}

.card_info_header {
    width: 100%;
    margin-top: 10px;
    line-height: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 16px;
}

.time {
    font-size: 15px;
    color: #999;
}

.card_info_middle {
    // margin-top: 8px;
    line-height: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.card_info_footer {
    line-height: 14px;
    display: flex;
    flex-direction: column;
}

.card_info_footer .seq_tags {
    display: flex;
    line-height: 14px;
    flex-wrap: wrap;
}

.classification_result {
    width: 100%;
    display: flex;
    flex-direction: column;
    margin-top: 8px;
}

.classification_result .el-text {
    font-size: 14px;
    color: #999;
}

.classification_result .category {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 3px;
    margin-bottom: 3px;
}

.show_button {
    padding: 2px 5px;
    min-width: auto;
    min-height: auto;
    color: #ffffff;
}

.delete_button {
    padding: 2px 5px;
    min-width: auto;
    min-height: auto;
    color: #ffffff;
}

.image {
    width: 90%;
    height: 80%;
    display: block;
}






.classification_result_main {
    width: 95%;
    height: 100%;
    border: #409EFF solid 1px;
}

.classification_result_content {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: row;
}

.classification_result_content_left {
    width: 50%;
    height: 95%;
    display: flex;
    flex-direction: column;
}

.classification_result_content_left_title {
    width: 100%;
    height: 10%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.classification_result_content_left_title span {
    font-size: 20px;
    font-weight: bold;
    padding-bottom: 10px;
}

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

.classification_result_content_left_img {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
}

.classification_result_content_left_img .classification_result_left_img {
    position: absolute;
    margin: auto;
}

.classification_result_content_left_img .classification_result_left_bottom_img {
    position: absolute;
    margin: auto;
}

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

.el-radio-group {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-left: 20px;
}

.classification_result_divider {
    height: 100%;
    margin: auto;
    display: flex;
    justify-content: center;
    align-items: center;
}

.classification_result_divider .el-divider {
    height: 90%;
}

.classification_result_content_right {
    width: 50%;
    height: 95%;
    display: flex;
    flex-direction: column;
}

.result_info {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.result_info .box-card {
    width: 60%;
    height: 55%;
    justify-content: space-around;
}

.result_info .box-card .info_icon {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    align-items: center;
}

.result_info .box-card .info_icon svg-icon {
    width: 100%;
    height: 100%;
}

.result_info .box-card .info_icon span {
    font-size: 25px;
    font-weight: bold;
}

.result_info .box-card .text {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.result_info .box-card .text span:nth-child(1) {
    font-size: 20px;
    margin-right: 10px;
}

.result_info .box-card .text span:nth-child(2) {
    font-size: 20px;
    font-weight: 500;
}

.result_info .box-card .el-button {
    width: 100%;
    height: 10%;
    margin-top: 120px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 10px;
    background-color: #409EFF;
    color: white;
    border: none;
}

.result_info .box-card .el-button:hover {
    background-color: #66b1ff;
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
}</style>
