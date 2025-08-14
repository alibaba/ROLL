import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ChatFloatButton from '../ChatFloatButton';

export default function AIChat() {
  const [url, setUrl] = useState('');

  const getUrl = async () => {
    try {
      const res = await axios.get('https://roll-dor-iframe-xfxnuxvngr.cn-shanghai.fcapp.run/get_signin_url');

      if (res?.data?.result) {
        setUrl(res.data.result);
      }
    } catch (error) {
      console.log("error", error);
    }
  }

  useEffect(() => {
    getUrl();
  }, []);

  if (!url) {
    return null;
  }

  return (
    <ChatFloatButton
      title="智能文档助手"
      content={
        <div style={{ width: '500px', height: '75vh' }}>
          <iframe src={url} frameborder="0" width="100%" height="100%"></iframe>
        </div>
      }
      floatButtonProps={{
        style: {
          insetInlineEnd: 24,
          bottom: 90,
          width: 48,
          height: 48,
        },
      }}
    />
  );
}