import datetime
import joblib
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path("model/joblib/model.joblib")


@st.cache_resource
def load_trained_model(model_path: Path):
    model = joblib.load(model_path)
    feature_names = None
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    return model, feature_names


def parse_work_year(work_year: str) -> float:
    if work_year == "< 1 year":
        return 0.0
    if work_year == "10+ years":
        return 10.0
    try:
        return float(work_year.split()[0])
    except Exception:
        return 0.0


def build_feature_dataframe(inputs: dict, feature_names: list[str] | None) -> pd.DataFrame:
    raw_df = pd.DataFrame([inputs])

    raw_df["issue_date"] = pd.to_datetime(raw_df["issue_date"], errors="coerce")
    raw_df["issue_year"] = raw_df["issue_date"].dt.year.fillna(0).astype(int)
    raw_df["issue_month"] = raw_df["issue_date"].dt.month.fillna(0).astype(int)
    raw_df = raw_df.drop(columns=["issue_date"])

    raw_df["work_year_num"] = raw_df["work_year"].apply(parse_work_year)
    raw_df = raw_df.drop(columns=["work_year"])

    raw_df["house_exist"] = int(raw_df["house_exist"].iloc[0])

    categorical_cols = ["class", "use", "region", "initial_list_status", "app_type"]
    for col in categorical_cols:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].astype(str)

    raw_df = pd.get_dummies(raw_df, columns=[col for col in categorical_cols if col in raw_df.columns], drop_first=True, dtype=int)

    if feature_names is not None:
        for feature in feature_names:
            if feature not in raw_df.columns:
                raw_df[feature] = 0
        raw_df = raw_df[feature_names]

    return raw_df


def main():
    st.set_page_config(
        page_title="信贷违约预测",
        page_icon="💳",
        layout="wide",
    )

    st.title("信贷违约预测系统")
    st.markdown(
        "通过已训练模型，预测借款人是否存在违约风险，并以可视化结果展示违约概率。"
    )

    loan_use_labels = {
        0: "其他/未分类",
        1: "教育贷款",
        2: "购车贷款",
        3: "购房贷款",
        4: "装修贷款",
        5: "医疗贷款",
        6: "婚嫁贷款",
        7: "旅游贷款",
        8: "经营贷款",
        9: "债务偿还贷款",
        10: "消费贷款",
        11: "投资贷款",
        12: "生活消费贷款",
        13: "专项贷款",
    }
    region_labels = {
        0: "北京",
        1: "天津",
        2: "河北",
        3: "山西",
        4: "内蒙古",
        5: "辽宁",
        6: "吉林",
        7: "黑龙江",
        8: "上海",
        9: "江苏",
        10: "浙江",
        11: "安徽",
        12: "福建",
        13: "江西",
        14: "山东",
        15: "河南",
        16: "湖北",
        17: "湖南",
        18: "广东",
        19: "广西",
        20: "海南",
        21: "重庆",
        22: "四川",
        23: "贵州",
        24: "云南",
        25: "西藏",
        26: "陕西",
        27: "甘肃",
        28: "青海",
        29: "宁夏",
        30: "新疆",
        31: "香港",
        32: "澳门",
        33: "台湾",
        34: "东北地区",
        35: "华北地区",
        36: "华东地区",
        37: "华南地区",
        38: "华中地区",
        39: "西南地区",
        40: "西北地区",
        41: "长三角地区",
        42: "珠三角地区",
        43: "京津冀地区",
        44: "环渤海地区",
        45: "成渝地区",
        46: "海峡西岸",
        47: "北部湾地区",
        48: "其他地区",
        49: "未知地区",
    }
    initial_list_status_labels = {
        0: "首发列表",
        1: "补发列表",
    }
    app_type_labels = {
        0: "个人申请",
        1: "联合申请",
    }

    st.sidebar.header("输入特征")
    with st.sidebar.form(key="input_form"):
        total_loan = st.text_input(
            "贷款总额（元）",
            value="20000.0",
            placeholder="",
            key="total_loan",
        )
        year_of_loan = st.selectbox(
            "贷款期限（年）",
            [3, 5],
            key="year_of_loan",
        )
        interest = st.text_input(
            "年利率 (%)",
            value="12.0",
            placeholder="",
            key="interest",
        )
        monthly_payment = st.text_input(
            "每月还款金额（元）",
            value="1200.0",
            placeholder="",
            key="monthly_payment",
        )
        work_year = st.selectbox(
            "工作年限",
            ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
            key="work_year",
        )
        house_exist = st.selectbox(
            "是否有房产",
            [0, 1],
            format_func=lambda x: "无" if x == 0 else "有",
            key="house_exist",
        )
        debt_loan_ratio = st.slider(
            "债务收入比", min_value=0.0, max_value=1.0, value=0.25, step=0.01, key="debt_loan_ratio"
        )
        del_in_18month = st.text_input(
            "18个月内逾期次数",
            value="0",
            placeholder="",
            key="del_in_18month",
        )
        scoring_low = st.text_input(
            "最低评分",
            value="600.0",
            placeholder="",
            key="scoring_low",
        )
        scoring_high = st.text_input(
            "最高评分",
            value="700.0",
            placeholder="",
            key="scoring_high",
        )
        loan_class = st.selectbox(
            "信用等级",
            ["A", "B", "C", "D", "E"],
            key="loan_class",
        )
        loan_use = st.selectbox(
            "贷款用途",
            list(loan_use_labels.keys()),
            format_func=lambda x: f"{x} - {loan_use_labels.get(x, '未知用途')}",
            key="loan_use",
        )
        region = st.selectbox(
            "地区",
            list(region_labels.keys()),
            format_func=lambda x: f"{x} - {region_labels.get(x, '未知地区')}",
            key="region",
        )
        initial_list_status = st.selectbox(
            "初始列表状态",
            list(initial_list_status_labels.keys()),
            format_func=lambda x: f"{x} - {initial_list_status_labels.get(x)}",
            key="initial_list_status",
        )
        app_type = st.selectbox(
            "申请类型",
            list(app_type_labels.keys()),
            format_func=lambda x: f"{x} - {app_type_labels.get(x)}",
            key="app_type",
        )
        issue_date = st.date_input(
            "发放日期",
            value=datetime.date.today(),
            key="issue_date",
        )

        st.markdown("---")
        submit_button = st.form_submit_button("开始预测")

    predict_button = submit_button

    if not MODEL_PATH.exists():
        st.warning("模型文件未找到，请先训练模型并生成 model/joblib/model.joblib。")
        return

    model, expected_features = load_trained_model(MODEL_PATH)

    # If we have a stored prediction from a prior run, show it. Otherwise, if the predict button
    # was just pressed, compute and store the result. If neither, show the initial prompt.
    if ("last_prediction" in st.session_state) and (not predict_button):
        # display stored result
        result_label = st.session_state.get("last_prediction")
        proba = st.session_state.get("last_proba", 0.0)
        probability_text = f"{proba:.2%}"

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("预测结论")
            if result_label == "违约":
                st.error(f"预测结果：{result_label}")
            else:
                st.success(f"预测结果：{result_label}")
            st.write("本次预测基于输入的信贷特征与已训练银行模型。")

        with col2:
            st.metric(label="违约概率", value=probability_text)
            st.metric(label="预测状态", value=result_label)

        st.markdown("---")
        st.subheader("风险等级可视化")
        progress_value = int(proba * 100)
        st.progress(progress_value)
        if proba >= 0.7:
            st.warning("违约风险较高，请谨慎审批。")
        elif proba >= 0.4:
            st.info("违约风险中等，可进一步核查用户信息。")
        else:
            st.success("违约风险较低，可继续推进审批流程。")

        st.markdown("---")
        st.subheader("输入摘要")
        summary_cols = st.columns(3)
        summary_cols[0].metric("贷款总额", f"{st.session_state.get('total_loan', 0):.0f} 元")
        summary_cols[1].metric("贷款期限", f"{st.session_state.get('year_of_loan', 0)} 年")
        summary_cols[2].metric("年利率", f"{st.session_state.get('interest', 0.0):.1f}%")

        summary_cols = st.columns(3)
        summary_cols[0].metric("还款金额", f"{st.session_state.get('monthly_payment', 0):.0f} 元")
        summary_cols[1].metric("债务收入比", f"{st.session_state.get('debt_loan_ratio', 0.0):.2f}")
        summary_cols[2].metric("信用等级", st.session_state.get('loan_class', ''))

    elif predict_button:
        try:
            total_loan_value = float(st.session_state.total_loan)
            interest_value = float(st.session_state.interest)
            monthly_payment_value = float(st.session_state.monthly_payment)
            del_in_18month_value = int(float(st.session_state.del_in_18month))
            scoring_low_value = float(st.session_state.scoring_low)
            scoring_high_value = float(st.session_state.scoring_high)
        except ValueError:
            st.error("请填写正确的数值输入，例如 20000、12.5、0 或 600。")
            return

        user_inputs = {
            "total_loan": total_loan_value,
            "year_of_loan": st.session_state.year_of_loan,
            "interest": interest_value,
            "monthly_payment": monthly_payment_value,
            "work_year": st.session_state.work_year,
            "house_exist": st.session_state.house_exist,
            "debt_loan_ratio": st.session_state.debt_loan_ratio,
            "del_in_18month": del_in_18month_value,
            "scoring_low": scoring_low_value,
            "scoring_high": scoring_high_value,
            "class": st.session_state.loan_class,
            "use": st.session_state.loan_use,
            "region": st.session_state.region,
            "initial_list_status": st.session_state.initial_list_status,
            "app_type": st.session_state.app_type,
            "issue_date": pd.to_datetime(st.session_state.issue_date),
        }
        feature_df = build_feature_dataframe(user_inputs, expected_features)

        try:
            prediction = model.predict(feature_df)[0]
            proba = float(model.predict_proba(feature_df)[0][1])
        except Exception:
            st.error("模型预测失败，请检查模型与输入特征格式是否一致。")
            return

        result_label = "违约" if prediction == 1 else "不违约"
        probability_text = f"{proba:.2%}"

        # save prediction to session so UI can persist until reset
        st.session_state["last_prediction"] = result_label
        st.session_state["last_proba"] = proba

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("预测结论")
            if prediction == 1:
                st.error(f"预测结果：{result_label}")
            else:
                st.success(f"预测结果：{result_label}")
            st.write("本次预测基于输入的信贷特征与已训练银行模型。")

        with col2:
            st.metric(label="违约概率", value=probability_text)
            st.metric(label="预测状态", value=result_label)

        st.markdown("---")
        st.subheader("风险等级可视化")
        progress_value = int(proba * 100)
        st.progress(progress_value)
        if proba >= 0.7:
            st.warning("违约风险较高，请谨慎审批。")
        elif proba >= 0.4:
            st.info("违约风险中等，可进一步核查用户信息。")
        else:
            st.success("违约风险较低，可继续推进审批流程。")

        st.markdown("---")
        st.subheader("输入摘要")
        summary_cols = st.columns(3)
        summary_cols[0].metric("贷款总额", f"{total_loan_value:.0f} 元")
        summary_cols[1].metric("贷款期限", f"{year_of_loan} 年")
        summary_cols[2].metric("年利率", f"{interest_value:.1f}%")

        summary_cols = st.columns(3)
        summary_cols[0].metric("还款金额", f"{monthly_payment_value:.0f} 元")
        summary_cols[1].metric("债务收入比", f"{debt_loan_ratio:.2f}")
        summary_cols[2].metric("信用等级", loan_class)
    else:
        st.info("请在侧边栏输入特征并点击“开始预测”以查看违约风险。")


if __name__ == "__main__":
    main()
