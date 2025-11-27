import React, { useState, useEffect } from 'react';
import { useColorMode } from '@docusaurus/theme-common';
import dayjs from 'dayjs';
import { Card, theme, ConfigProvider, DatePicker } from 'antd';
import EchartsView from '../Echarts';
import locale from 'antd/locale/zh_CN';

import 'dayjs/locale/zh-cn';

dayjs.locale('zh-cn');

const { RangePicker } = DatePicker;

export default ({ allStat }) => {
  const { colorMode } = useColorMode();
  const today = dayjs().format('YYYY-MM-DD');
  const defaultDates = [dayjs().subtract(7, 'day'), dayjs()];
  const [dates, setDates] = useState(defaultDates);
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    // 根据dates的值，从allStat里获取到最终的数组，组装成按照日期从小到大的数组
    if (dates && dates.length === 2 && dates[0] && dates[1]) {
      const startDate = dayjs(dates[0]);
      const endDate = dayjs(dates[1]);

      // 生成起止日期之间的所有日期
      const dateArray = [];
      let currentDate = startDate.clone();
      while (currentDate.isBefore(endDate) || currentDate.isSame(endDate)) {
        dateArray.push(currentDate.format('YYYY-MM-DD'));
        currentDate = currentDate.add(1, 'day');
      }

      // 从allStat中提取对应日期的数据
      const filteredData = dateArray.map(date => ({
        date,
        ...(allStat[date] || {})
      })).filter(item => item.date); // 过滤掉没有日期的数据

      setChartData(filteredData);
    }
  }, [dates, allStat]);

  return <ConfigProvider locale={locale} theme={{ algorithm: colorMode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}>
    <div style={{ width: '80%', margin: '0 auto', marginTop: 20 }}>
      <RangePicker defaultValue={defaultDates} onChange={dates => setDates(dates)} presets={[{ label: '最近7天', value: [dayjs().subtract(7, 'day'), dayjs()] }, { label: '最近30天', value: [dayjs().subtract(30, 'day'), dayjs()] }]} />
      <Card title={`${dates.map(item => item.format('YYYY-MM-DD')).join('——')}数据统计图`}>
        <EchartsView
          option={{
            tooltip: {
              trigger: 'axis'
            },
            legend: {
              data: ['Fork数', 'Star数', 'Contributors', 'issues总数', 'issues open数', 'issue 解决率', 'PR总数', 'PR open数'],
              bottom: 20,
            },
            xAxis: {
              type: 'category',
              boundaryGap: false,
              data: chartData.map(item => item.date),
            },
            yAxis: {
              type: 'value'
            },
            series: [
              {
                name: 'Fork数',
                type: 'line',
                data: chartData.map(item => item.forks || 0)
              },
              {
                name: 'Star数',
                type: 'line',
                data: chartData.map(item => item.stars || 0)
              },
              {
                name: 'Contributors',
                type: 'line',
                data: chartData.map(item => item.contributors || 0)
              },
              {
                name: 'issues总数',
                type: 'line',
                data: chartData.map(item => item.issues?.total || 0)
              },
              {
                name: 'issues open数',
                type: 'line',
                data: chartData.map(item => item.issues?.open || 0)
              },
              {
                name: 'issue 解决率',
                type: 'line',
                data: chartData.map(item => item.issues?.fixRate || 0)
              },
              {
                name: 'PR总数',
                type: 'line',
                data: chartData.map(item => item.prs?.total || 0)
              },
              {
                name: 'PR open数',
                type: 'line',
                data: chartData.map(item => item.prs?.open || 0)
              },
            ],
          }}
          style={{ widht: '100%', height: 400 }}
        ></EchartsView>
      </Card>
    </div>
  </ConfigProvider>
}