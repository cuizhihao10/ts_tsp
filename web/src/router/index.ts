import { createWebHistory, createRouter } from "vue-router";
import { constantRoute } from './routes';
import { useTokenStore } from "@/store/modules/mytoken";
import useUserInfoStore from '@/store/modules/userInfo'


const router = createRouter({
    // 这里使用历史记录模式
    history: createWebHistory(),
    routes: constantRoute,
	scrollBehavior() {
		return {
			left: 0,
			top: 0,
		}
	}
});

router.beforeEach((to, from, next) => {	
	if (to.matched.some(record => record.meta?.requiresAuth)) {
		const store = useTokenStore();
		const userInfoStore = useUserInfoStore();
		if (!store.token.access_token && !userInfoStore.getuserInfo.logined) {
			next({ name: 'login', query: { redirect: to.fullPath }});
			return;
		}
	}
	next();
});

export default router;