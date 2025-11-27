// 引入 echarts 核心模块，核心模块提供了 echarts 使用必须要的接口。
import * as echarts from 'echarts/core';
// 引入柱状图图表，图表后缀都为 Chart
import { LineChart } from 'echarts/charts';
// 引入标题，提示框，直角坐标系，数据集，内置数据转换器组件，组件后缀都为 Component
import {
  DatasetComponent,
  DataZoomComponent,
  GraphicComponent,
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
  TransformComponent,
  MarkLineComponent,
  MarkPointComponent,
} from 'echarts/components';
// 标签自动布局、全局过渡动画等特性

import { LabelLayout, UniversalTransition } from 'echarts/features';
// 引入 Canvas 渲染器，注意引入 CanvasRenderer 或者 SVGRenderer 是必须的一步
import { CanvasRenderer } from 'echarts/renderers';
import { useColorMode } from '@docusaurus/theme-common';

// 注册必须的组件
echarts.use([
  LineChart,
  GraphicComponent,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DatasetComponent,
  TransformComponent,
  LegendComponent,
  LabelLayout,
  UniversalTransition,
  CanvasRenderer,
  DataZoomComponent,
  MarkLineComponent,
  MarkPointComponent,
]);

import { useEffect, useRef } from 'react';

const EchartsView = (props) => {
  const { option, className, style, onEvents = {}, onInit } = props;
  const echartsRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const { colorMode } = useColorMode();

  useEffect(() => {
    // 确保 DOM 元素存在且有尺寸
    if (!echartsRef.current) return;
    // 检查容器是否有尺寸
    const container = echartsRef.current;
    if (container.clientWidth === 0 || container.clientHeight === 0) {
      // 如果容器没有尺寸，使用 ResizeObserver 监听尺寸变化
      const resizeObserver = new ResizeObserver((entries) => {
        for (let entry of entries) {
          if (entry.contentRect.width > 0 && entry.contentRect.height > 0) {
            // 容器有尺寸了，初始化图表
            initializeChart();
            resizeObserver.disconnect();
            break;
          }
        }
      });

      resizeObserver.observe(container);
      // 添加一个超时机制作为后备方案
      const timeoutId = setTimeout(() => {
        resizeObserver.disconnect();
        if (container.clientWidth > 0 || container.clientHeight > 0) {
          initializeChart();
        }
      }, 100);

      return () => {
        resizeObserver.disconnect();
        clearTimeout(timeoutId);
      };
    } else {
      // 容器已经有尺寸，直接初始化
      initializeChart();
    }
    function initializeChart() {
      // 如果已经有图表实例，先销毁
      if (chartInstanceRef.current) {
        chartInstanceRef.current.dispose();
      }

      // 初始化图表
      chartInstanceRef.current = echarts.init(echartsRef.current, colorMode);
      if (!chartInstanceRef.current) return;

      chartInstanceRef.current.setOption(option);

      if (onInit && typeof onInit === 'function') {
        onInit(chartInstanceRef.current);
      }
      // 绑定事件
      Object.keys(onEvents).forEach((eventName) => {
        chartInstanceRef.current.on(eventName, (params) => {
          onEvents[eventName](params, chartInstanceRef.current);
        });
      });
    }

    // 添加窗口变化事件监听器
    const resizeHandler = () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.resize();
      }
    };

    window.addEventListener('resize', resizeHandler);

    return () => {
      // 移除所有事件监听器
      Object.keys(onEvents).forEach((eventName) => {
        if (chartInstanceRef.current) {
          chartInstanceRef.current.off(eventName);
        }
      });
      // 移除窗口变化事件监听器
      window.removeEventListener('resize', resizeHandler);

      // 销毁图表实例
      if (chartInstanceRef.current) {
        chartInstanceRef.current.dispose();
        chartInstanceRef.current = null;
      }
    };
  }, [option, colorMode]);

  return <div ref={echartsRef} className={className} style={style}></div>;
};

export default EchartsView;
