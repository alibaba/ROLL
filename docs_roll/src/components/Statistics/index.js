import React from 'react';
import { useColorMode } from '@docusaurus/theme-common';
import dayjs from 'dayjs';
import { Card, Statistic, theme, Flex, ConfigProvider } from 'antd';

export default ({ todayStat }) => {
  const { colorMode } = useColorMode();
  const today = dayjs().format('YYYY-MM-DD');

  return <ConfigProvider theme={{ algorithm: colorMode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}>
    <div style={{ width: '80%', margin: '0 auto', marginTop: 20 }}>
      <Card title={`${today}数据统计`}>
        <Flex wrap justify="space-between">
          <Statistic title="star数" value={todayStat?.stars} />
          <Statistic title="fork数" value={todayStat?.forks} />
          <Statistic title="contributors" value={todayStat?.contributors} />
          <Statistic title="issues总数" value={todayStat?.issues?.total} />
          <Statistic title="issues open数" value={todayStat?.issues?.open} />
          <Statistic title="issues 解决率" value={`${todayStat?.issues?.fixRate || 0}%`} />
          <Statistic title="PR总数" value={todayStat?.prs?.total} />
          <Statistic title="PR open数" value={todayStat?.prs?.open} />
        </Flex>
      </Card>
    </div>
  </ConfigProvider>
}