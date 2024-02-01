export const constantRoute = [
    {
        path: '/login/',
        name: 'login',
        component: () => import('@/views/login/LoginIndexView.vue')
    },
    {
        path: '/',
        name: 'layout',
        component: () => import('@/layout/LayoutIndexView.vue'),
        meta: { requiresAuth: true },
        redirect: '/home/',
        children: [
            {
                path: 'home/',
                name: 'Home',
                component: () => import('@/views/home/HomeIndexView.vue'),
                children: [
                    {
                        path: 'upload/',
                        name: 'Upload',
                        component: () => import('@/views/segmentation/SegmentationIndexView.vue')
                    }
                ]
            },
            // {
            //     path: '/segmentation/',
            //     name: 'Segmentation',
            //     component: () => import('@/views/segmentation/SegmentationIndexView.vue'),
            // },
            {
                path: '/segmentation_history/',
                name: 'SegmentationHistory',
                component: () => import('@/views/segmentationhistory/SegmentationHistoryIndexView.vue'),
            },
            {
                path: '/classification/',
                name: 'Classification',
                component: () => import('@/views/classification/ClassificationIndexView.vue'),
            },
            {
                path: '/classification_history/',
                name: 'ClassificationHistory',
                component: () => import('@/views/classificationhistory/ClassificationHistoryIndexView.vue'),
            }
        ]
    },
    {
        path: '/404/',
        name: 'NotFound',
        meta: {
            title: 'Page Not Found'
        },
        component: () => import('@/views/error/NotFoundView.vue')
    },
    // 所有未定义路由，全部重定向到 404
    {
        path: '/:pathMatch(.*)*',
        redirect: '/404/'
    }
];

// 获取所有路由
