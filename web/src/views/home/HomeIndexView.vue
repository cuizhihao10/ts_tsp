<template>
    <el-row v-if="isRouter">
        <el-col class="task_card">
            <el-card class="box-card" shadow="hover" :body-style="{ padding: '0px' }" @mouseleave="visible = !visible"
                @mouseenter="visible = !visible">
                <img src="@/assets/pictures/分割.png" class="image" />
                <div style="padding: 14px; font-size: 30px;">
                    <span>{{ tasks[0].name }}</span>
                    <div class="bottom">
                        <div class="info_all">
                            <span class="info">{{ tasks[0].description[1] }}</span>
                            <span class="info">{{ tasks[0].description[2] }}</span>
                        </div>
                        <n-button style="width: 150px; height: 42px; font-size: 20px;" type="primary" round
                            @click="goPage(tasks[0].path)">
                            Start Task
                        </n-button>
                    </div>
                </div>
                <transition name="slide">
                    <div class="card-pop-menu" v-show="visible">
                        <!-- <n-radio-group v-model:value="myEffect" style="margin-bottom: 10px">
                                <n-radio-button v-for="effect in effects" :key="effect" :value="effect">
                                    {{ effect }}
                                </n-radio-button>
                            </n-radio-group> -->
                        <n-carousel :key="myEffect" :centered-slides="isCard" :slides-per-view="isCard ? 'auto' : 1"
                            draggable style="height: 40vh">
                            <n-carousel-item :style="{ width: isCard ? '85%' : '100%' }">
                                <img class="carousel-img" src="@/assets/CarouselFigure/input.jpg">
                            </n-carousel-item>
                            <n-carousel-item :style="{ width: isCard ? '85%' : '100%' }">
                                <img class="carousel-img" src="@/assets/CarouselFigure/model.png">
                            </n-carousel-item>
                            <n-carousel-item :style="{ width: isCard ? '85%' : '100%' }">
                                <img class="carousel-img" src="@/assets/CarouselFigure/output.jpg">
                            </n-carousel-item>
                        </n-carousel>
                    </div>
                </transition>
            </el-card>
        </el-col>
    </el-row>
    <router-view v-if="!isRouter"></router-view>
</template>

<script setup lang='ts'>
import router from '@/router';
const isRouter = ref(true);
const visible = ref(false)
const myEffect = ref('card')
const isCard = computed(() => myEffect.value === 'card')
const tasks = ref([
    {
        name: 'Brain MRI analysis',
        path: '/home/upload/',
        description: {
            1: '1.Segmenting tumors in MRI images',
            2: '2.Classifying glioma subtypes'
        },
    },
])
// const isCard = isCardRef
// const myEffect = effectRef
// const effects = ['slide', 'fade', 'card']
watch(() => router.currentRoute.value, (newValue: any) => {
    if (newValue.path === '/home/') {
        isRouter.value = true;
    } else {
        isRouter.value = false;
    }
},
    { immediate: true }
)

const goPage = (path: string) => {
    isRouter.value = false;
    router.push({ path: path })
}

</script>

<style lang="scss" scoped>
.task_card {
    display: flex;
    align-items: center;
}

.box-card {
    display: flex;
    flex-direction: column;
    width: 30vw;
    height: 75vh;
    font-family: serif;
    font-size: 20px;
    font-weight: 600;
}


.card-pop-menu {
    border-width: 1px 0 0 0;
    position: relative;
    top: -50vh;
}

.carousel-img {
    margin: 0 auto;
    width: 100%;
    height: 100%;
    object-fit: fill;
}

.slide-enter-active {
    transition: all 0.2s linear;
}

.slide-leave-active {
    transition: all 0.2s linear;
}

.slide-enter-from,

.slide-leave-to {
    transform: translateY(88px);
}

.info_all {
    display: flex;
    flex-direction: column;
}

.info {
    font-size: 20px;
    margin-bottom: 10px;
    line-height: 25px;
    color: #999;
}

.bottom {
    margin-top: 13px;
    line-height: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.button {
    padding: 0;
    min-height: auto;
}

.image {
    width: 100%;
    display: block;
}
</style>