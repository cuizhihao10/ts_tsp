<template>
    <div class="content">
        <div class="main">
            <div class="search">
                <el-autocomplete class="input" v-model="state" :fetch-suggestions="querySearchAsync"
                    placeholder="Please input taskname or time such as 'Test' or '2021-01-01'" @select="handleSelect"
                    style="width: 40%;">
                    <template #prepend>
                        <el-select v-model="select" placeholder="Select" style="width: 130px; height: 100%;" size="large"
                            font-size="18px">
                            <el-option label="Task name" value="task_name" />
                            <el-option label="Create Time" value="create_time" />
                        </el-select>
                    </template>
                    <template #append>
                        <el-button type="primary" style="display: flex;" @click="search">
                            <el-icon style="vertical-align: middle" color="#409EFC">
                                <Search />
                            </el-icon>
                            <span style="vertical-align: middle" font-size="18px"> Search </span>
                        </el-button>
                    </template>
                </el-autocomplete>
                <span>{{ state }}</span>
                <el-button class="order_button" type="primary" style="width: 5%; margin-left: 10px; height: 30%;" @click="SwitchOrder">Reverse order
                    <div class="iconBox">
                        <el-icon style="height: 10px" :color='upState ? "black" : ""'>
                            <CaretTop style="width: 15px; height: 10px; top: 4px" />
                        </el-icon>
                        <el-icon style="height: 10px" :color='downState ? "black" : ""'>
                            <CaretBottom style="width: 15px; height: 10px" />
                        </el-icon>
                    </div>
                </el-button>
                <el-button class="reset_button" style="width: 5%; height: 30%; margin-left: 10px;">Reset</el-button>
            </div>
            <el-row>
                <el-col v-for="(o, index) in history_tasks.length" :key="o" :span="6" :offset="index > 0 ? 1 : 1">
                    <el-card :body-style="{ padding: '10px', width: 'auto', height: '92%' }">
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
                            <time class="time">{{ history_tasks[index]['segmentation_time'] }}</time>
                            <div class="options">
                                <el-button size="large" dark color="#28ca61" :icon="Search" circle class="show_button"
                                    @click="openDialog(history_tasks[index])"></el-button>
                                <el-button size="large" dark color="#d64141" :icon="Delete" circle class="delete_button"
                                    @click="delete_task(history_tasks[index])"></el-button>
                            </div>
                        </div>
                        <div class="card_info_footer my-2 flex flex-wrap gap-1 items-center">
                            <el-tag v-for="sequence in get_sequence(index)" :key="sequence.label" class="mx-1"
                                :type="sequence.type" effect="light">
                                {{ sequence.label }}
                            </el-tag>
                        </div>
                    </el-card>
                </el-col>
            </el-row>
        </div>
        <div class="footer">
            <el-pagination background layout="prev, pager, next, total" @current-change="handleCurrentChange"
                :page-size="pagination_page_size" :current-page="pagination_page_number" :total="pagination_total" />
        </div>
    </div>
    <el-dialog v-model="showModalRef" @closed="handleDialogClosed" width="80%" :title="modal_title">
        <el-form class="modal_main" v-loading="fullscreenLoading"
            element-loading-text="Inferring model in progress, please wait..." :element-loading-spinner="svg"
            element-loading-background="rgba(242,250,255, 0.7)" element-loading-svg-view-box="-10, -10, 50, 50">
            <div class="segmentation_result_main">
                <div class="segmentation_result_content">
                    <div class="segmentation_result_content_left">
                        <div class="segmentation_result_content_left_title">
                            <span>Original image</span>
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
                    </div>
                    <div class="segmentation_result_divider">
                        <el-divider direction="vertical" />
                    </div>
                    <div class="segmentation_result_content_right">
                        <div class="segmentation_result_content_right_title">
                            <span>Segmentation results</span>
                        </div>
                        <div class="radios">
                            <span>Sequence</span>
                            <el-radio-group v-model="radio2" v-for="sequence in original_sequence_list" :key="sequence"
                                @change="radio_change">
                                <el-radio :label="sequence" :key="sequence" border>{{ sequence }}</el-radio>
                                {{ sequence }}
                            </el-radio-group>
                        </div>
                        <div class="Luminance-slider-block">
                            <span>Luminance</span>
                            <el-slider v-model="seg_luminance" @input="drawRightImage(radio2)" :min="-100" />
                        </div>
                        <div class="segmentation_result_content_right_img">
                            <canvas id="right_bottom_images" class="segmentation_result_right_bottom_img"></canvas>
                            <canvas id="right_img" class="segmentation_result_right_img"
                                :style="{ opacity: transparency }"></canvas>
                            <div class="vertical_slider-block">
                                <el-slider v-model="transparency" @input="drawRightImage(radio2)"
                                    :format-tooltip="formatTooltip" vertical />
                                <span>Transparency</span>
                            </div>
                        </div>
                        <div class="slider-block">
                            <span>Serial Number</span>
                            <el-slider v-model="seg_number" :max="seg_number_max" @input="drawRightImage(radio2)"
                                show-input />
                        </div>
                    </div>
                </div>
                <div class="segmentation_radios_result">
                    <div class="info_icon">
                        <svg-icon name="结果对比分析" width="150px" height="150px"></svg-icon>
                        <div claas="info_info">
                            <span class="info_title">Segmentation Results<br></span>
                            <span class="info_p">(Proportion of tumors in each region)</span>
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
import { onMounted, ref } from 'vue'
import { inject } from "vue";
import {
    Delete,
    Search,
    CaretTop,
    CaretBottom,
} from '@element-plus/icons-vue'
import { seg_history, search_autocomplete, seg_history_search, seg_history_detail, seg_history_delete } from '@/api/history'
import useUserInfoStore from '@/store/modules/userInfo';


interface Item {
    type: string;
    label: string;
}

const items = ref<Array<Item>>([
    { type: '', label: 'Flair' },
    { type: 'success', label: 'T1' },
    { type: 'info', label: 'T2' },
    { type: 'danger', label: 'T1ce' },
])

const state = ref('')
const select = ref('')
interface SearchItem {
    task_name: string
    create_time: string
}

const search_results = ref<SearchItem[]>([])
let timeout: ReturnType<typeof setTimeout>
let upState = ref(false);
let downState = ref(false);

const showModalRef = ref(false);
const userInfoStore = useUserInfoStore();
const history_tasks = ref([]);
const modal_original_pic = ref({
    src: '../../public/',
    pic_number: 0
});
const modal_segmented_pic = ref({
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
const radio2 = ref<string>('flair');
const pagination_total = ref(0);
const pagination_page_size = ref(6);
const pagination_page_number = ref(1);


type ori_pic = {
    [key: string]: string[]
}
type ori_pic_num = {
    [key: string]: number
}
const original_sequence_list = ref<string[]>([])
const original_pic = ref<ori_pic>({})
const original_number = ref<number>(0)
const original_number_max = ref<ori_pic_num>({})
const seg_pic = ref<string[]>([])
const seg_number = ref<number>(0)
const seg_number_max = ref<number>(0)
const ori_luminance = ref<number>(0)
const seg_luminance = ref<number>(0)
const transparency = ref<number>(50)
const NCR_ratio = ref<number>(0)
const ED_ratio = ref<number>(0)
const ET_ratio = ref<number>(0)
const reload = inject("reload");

onMounted(async () => {
    await seg_history({ userId: userInfoStore.getuserInfo.uid, page: pagination_page_number.value, page_size: pagination_page_size.value }).then(res => {
        if (res.data.code == 200) {
            console.log(res.data.data)
            history_tasks.value = res.data.data.page_list
            pagination_total.value = res.data.data.total
        } else {
            console.log('获取任务失败')
        }
    })
    await loadAll()
})

let SwitchOrder = () => {
    // upState.value = !upState.value;
    downState.value = !downState.value;
}

const search = async () => {
    console.log({select: select.value, state: state.value})
    await seg_history_search({ userId: userInfoStore.getuserInfo.uid, page: pagination_page_number.value, page_size: pagination_page_size.value, select: select.value, state: state.value }).then(res => {
        if (res.data.code == 200) {
            console.log(res.data.data)
            history_tasks.value = res.data.data.page_list
            pagination_total.value = res.data.data.total
        } else {
            console.log('获取任务失败')
        }
    })
}

const handleCurrentChange = async (current_page: number) => {
    pagination_page_number.value = current_page
    console.log(pagination_page_number.value)
    await seg_history({ userId: userInfoStore.getuserInfo.uid, page: pagination_page_number.value, page_size: pagination_page_size.value }).then(res => {
        if (res.data.code == 200) {
            console.log(res.data.data)
            history_tasks.value = res.data.data.page_list
            pagination_total.value = res.data.data.total
        } else {
            console.log('获取任务失败')
        }
    })
}


const loadAll = async () => {
    await search_autocomplete({ userId: userInfoStore.getuserInfo.uid }).then(res => {
        if (res.data.code == 200) {
            console.log(res.data.data)
            search_results.value = res.data.data.task_list
        } else {
            console.log('获取任务失败')
        }
    })
}

const querySearchAsync = (queryString: string, cb: (arg: any) => void) => {
    const results = queryString
        ? search_results.value.filter(createFilter(queryString))
        : search_results.value
    console.log(results)
    clearTimeout(timeout)
    timeout = setTimeout(() => {
        cb(results)
    }, 5000 * Math.random())
}

const createFilter = (queryString: string) => {
    if (select.value === 'create_time')
        return (search_results: SearchItem) => {
            
            return (
                search_results.create_time.toLowerCase().includes(queryString.toLowerCase())
            )
        }
    else if (select.value === 'task_name')
        return (search_results: SearchItem) => {
            console.log(search_results.task_name)
            return (
                search_results.task_name.toLowerCase().indexOf(queryString.toLowerCase()) === 0
            )
        }
    else
        return (search_results: SearchItem) => {
            return (
                search_results.task_name.toLowerCase().indexOf(queryString.toLowerCase()) === 0
            )
        }
}

const handleSelect = (item: Record<string, any>) => {
    console.log(item)
}

onMounted(() => {
    // links.value = loadAll()
})

const radio_change = () => {
    drawLeftImage(radio1.value)
    drawRightImage(radio2.value)
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
    return "http://127.0.0.1:5000/" + history_tasks.value[index]['original_images_path'] + "/100.png"
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
                    await seg_history_delete({ userId: userInfoStore.getuserInfo.uid, task_name: task.filename })
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
    await seg_history_detail({ userId: userInfoStore.getuserInfo.uid, task_name: task.filename }).then((res: any) => {
        if (res.data.code == 200) {
            console.log(res.data.data)
            fullscreenLoading.value = false
            original_sequence_list.value = res.data.data.original_sequence_list
            original_pic.value = res.data.data.original_images
            original_number_max.value = res.data.data.original_num
            seg_pic.value = res.data.data.seg_images
            seg_number_max.value = res.data.data.seg_num
            NCR_ratio.value = JSON.parse(res.data.data.segmentation_result)['NCR_ratio']
            ED_ratio.value = JSON.parse(res.data.data.segmentation_result)['ED_ratio']
            ET_ratio.value = JSON.parse(res.data.data.segmentation_result)['ET_ratio']
            modal_title.value = task.filename
            drawLeftImage(radio1.value)
            drawRightImage(radio2.value)

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
        modal_segmented_pic.value.src = '/',
        modal_segmented_pic.value.pic_number = 0
}

const formatTooltip = (val: number) => {
    return val / 100
}

const drawLeftImage = (sequence: string) => {
    const canvas = document.getElementById("left_img") as HTMLCanvasElement;
    const ctx = canvas!.getContext("2d");
    const img = new Image();
    img.src = original_pic.value[sequence][original_number.value];
    img.onload = () => {
        canvas!.width = img.width;//设置画布大小和图片相同，否则画布有默认大小
        canvas!.height = img.height;

        ctx!.drawImage(img, 0, 0, img.width, img.height);//画图片
        const imgData = ctx!.getImageData(0, 0, img.width, img.height);//获取图片数据
        const data = imgData.data;//获取图片数据的data部分

        for (var i = 0; i < data.length; i += 4) {
            data[i + 0] += ori_luminance.value // r，红通道
            data[i + 1] += ori_luminance.value // g，绿通道
            data[i + 2] += ori_luminance.value // b，蓝通道
        }
        ctx!.clearRect(0, 0, canvas!.width, canvas!.width);//清除画布
        ctx!.putImageData(imgData, 0, 0);//重新绘制图片数据
    }
}

const right_bottom_images = (sequence: string) => {
    const canvas = document.getElementById("right_bottom_images") as HTMLCanvasElement;
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

const drawRightImage = (sequence: string) => {
    const canvas = document.getElementById("right_img") as HTMLCanvasElement;
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

    right_bottom_images(sequence)
}

</script>

<style lang="scss" scoped>
.content {
    width: 100%;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.main {
    width: 90%;
    height: 100%;
    display: flex;
    flex-direction: column;
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

.search {
    width: 100%;
    height: 10%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.search {
    :deep(.el-input__inner) {
        font-size: 18px;
        /* 你想要的字体大小 */
    }
}

// .order_button:hover {
//     background: #ecf5ff;
//     border-color: #c6e2ff;
//     color: #409eff;
//     display: flex;
//     justify-content: center;
//     align-items: center;
// }

.order_button .iconBox {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.reset_button {
    background: #fff;
    border-color: #cbcbcd;
    color: #505255;
}

.modal_main {
    height: 120vh;
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
    align-items: center;
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






.segmentation_result_main {
    width: 95%;
    height: 100%;
    border: #409EFF solid 1px;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.segmentation_result_content {
    width: 100%;
    height: 74%;
    display: flex;
    flex-direction: row;
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

.segmentation_result_divider {
    height: 100%;
    margin: auto;
    display: flex;
    justify-content: center;
    align-items: center;
}

.segmentation_result_divider .el-divider {
    height: 90%;
}

.segmentation_result_content_right {
    width: 50%;
    height: 95%;
    display: flex;
    flex-direction: column;
}

.segmentation_result_content_right_title {
    width: 100%;
    height: 10%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.segmentation_result_content_right_title span {
    font-size: 20px;
    font-weight: bold;
}

.el-radio-group {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-left: 20px;
}

.segmentation_result_content_right_img {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
}

.segmentation_result_content_right_img .segmentation_result_right_img {
    position: absolute;
    margin: auto;
}

.segmentation_result_content_right_img .segmentation_result_right_bottom_img {
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

.segmentation_radios_result {
    width: 60%;
    height: 25%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: row;
    background-color: rgb(147, 252, 243);
    border-radius: 10px;
    box-shadow: 0 0 15px rgba(0, 0, 0, 0.35);
}

.segmentation_radios_result .info_icon {
    width: 30%;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.segmentation_radios_result .info_icon svg-icon {
    width: 100%;
    height: 100%;
}

.segmentation_radios_result .info_icon .info_info {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}

.segmentation_radios_result .info_icon .info_title {
    font-size: 30px;
    font-weight: bold;
    color: #000;
    display: flex;
    text-align: center;
}

.segmentation_radios_result .info_icon .info_p {
    font-size: 18px;
    font-weight: normal;
    display: flex;
    text-align: center;
}

.segmentation_radios_result .divider {
    width: 2.5px;
    height: 80%;
    background-color: rgb(255, 255, 255);
    margin-left: 20px;
    margin-right: 20px;
}

.segmentation_radios_result .ratio_info {
    width: 50%;
    height: 100%;
    font-size: 20px;
    font-weight: bold;
    color: #000;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.segmentation_radios_result .ratio_info .text {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
</style>
