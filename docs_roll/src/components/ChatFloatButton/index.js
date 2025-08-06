import React, { useState } from 'react';
import { Popover, FloatButton } from 'antd';
import { CloseOutlined, CommentOutlined } from '@ant-design/icons';
import style from './styles.module.css';

export default ({ title, content, floatButtonProps = {} }) => {
  const [open, setOpen] = useState(false);

  return (
    <Popover
      placement="topRight"
      title={
        <div className={style.header}>
          <div>{title}</div>
          <CloseOutlined onClick={() => { setOpen(false); }} style={{ padding: '2px' }} />
        </div>
      }
      content={content}
      overlayClassName={style.floatButtonOverlay}
      open={open}
    >
      <FloatButton
        shape="circle"
        type="primary"
        className={style.floatButton}
        style={{ insetInlineEnd: 94 }}
        icon={<CommentOutlined style={{ fontSize: '24px' }} />}
        onClick={() => {
          setOpen(!open);
        }}
        tooltip="点击使用智能文档助手"
        {...floatButtonProps}
      />
    </Popover>
  );
};
