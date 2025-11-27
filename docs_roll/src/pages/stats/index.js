import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import dayjs from 'dayjs';
import Statistics from '../../components/Statistics';
import StatChart from '../../components/StatChart';

export default () => {
  const [todayStat, setTodayStat] = useState({});
  const [allStat, setAllStat] = useState({});
  const today = dayjs().format('YYYY-MM-DD');
  useEffect(() => {
    fetch('/ROLL/stats.json').then(res => res.json()).then(data => {
      setTodayStat(data[today]);
      setAllStat(data);
    })
  }, []);

  return <Layout>
    <main>
      <div>
        <Statistics todayStat={todayStat} />
        <StatChart allStat={allStat}></StatChart>
        <div style={{ width: '80%', margin: '0 auto', marginTop: '20px' }}>
          文档页面详细统计可以打开<a target='_blank' href="https://analytics.google.com/analytics/web/#/a375760323p513871422/reports/reportinghub?params=_u..nav%3Dmaui%26_u.dateOption%3Dyesterday%26_u.comparisonOption%3Ddisabled">Google analytics</a>查看
        </div>
      </div>
    </main>
  </Layout >
}