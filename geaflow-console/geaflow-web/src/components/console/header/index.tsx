/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/* eslint-disable no-param-reassign */
/* eslint-disable @typescript-eslint/no-unused-vars */
import React, { useEffect, useState } from "react";
import {
  DownOutlined,
  GlobalOutlined,
  UserSwitchOutlined,
  LogoutOutlined,
} from "@ant-design/icons";
import { Button, Dropdown, Space, Menu, Tag, message } from "antd";
import cx from "classnames";
import { useTranslation } from "react-i18next";
import cls from "./index.less";
import { useAuth } from "../hooks/useAuth";
import { queryInstanceList } from "../services/instance";
import { switchUserRole } from "../services/quickInstall";
import { isNull } from "lodash";
import { history } from "umi";
import i18n from "@/components/i18n";

interface PluginPorps {
  redirectPath?: RedirectPath[];
}

interface RedirectPath {
  path: string;
  pathName: string;
}

export const GeaflowHeader: React.FC<PluginPorps> = ({ redirectPath }) => {
  const redirectUrl = "/";

  const { t } = useTranslation();

  const isAdminLogin = localStorage.getItem("IS_GEAFLOW_ADMIN");
  const [state, setState] = useState({
    instanceList: [],
    isAdminLogin,
    currentInstance: isAdminLogin
      ? []
      : localStorage.getItem("GEAFLOW_CURRENT_INSTANCE")
      ? JSON.parse(localStorage.getItem("GEAFLOW_CURRENT_INSTANCE"))
      : null,
  });
  localStorage.setItem(
    "IS_ADMIN_LOGIN",
    isNull(state.isAdminLogin) ? "add" : ""
  );

  const { onLogout } = useAuth();
  const handleLogout = () => {
    onLogout().then((res) => {
      if (res.code === "SUCCESS") {
        localStorage.removeItem("GEAFLOW_LOGIN_USERNAME");
        localStorage.removeItem("QUICK_INSTALL_PARAMS");
        localStorage.removeItem("GEAFLOW_CURRENT_INSTANCE");
        localStorage.removeItem("GEAFLOW_TOKEN");
        localStorage.removeItem("IS_GEAFLOW_ADMIN");
        localStorage.removeItem("HAS_EXEC_QUICK_INSTALL");
        history.push(redirectUrl);
      } else {
        if (res.code === "FORBIDDEN") {
          history.push("/login");
          return;
        }
      }
    });
  };

  const getInstanceList = async () => {
    const resp = await queryInstanceList();
    // 如果没有登录或没有权限，直接跳转到登录页面
    if (!resp || resp.code === "FORBIDDEN") {
      history.push("/login");
      return;
    }

    if (resp.code === "SUCCESS") {
      // 是否存在默认的 Instance
      const defaultSelectInstance = localStorage.getItem(
        "GEAFLOW_CURRENT_INSTANCE"
      );
      if (!defaultSelectInstance) {
        const defaultInstance = resp.data?.list[0];
        if (defaultInstance) {
          localStorage.setItem(
            "GEAFLOW_CURRENT_INSTANCE",
            JSON.stringify({
              key: defaultInstance.id,
              value: defaultInstance.name,
            })
          );
          setState({
            ...state,
            instanceList: resp.data?.list,
            currentInstance: {
              key: defaultInstance.id,
              value: defaultInstance.name,
            },
          });
        }
      } else {
        setState({
          ...state,
          instanceList: resp.data?.list,
          currentInstance: JSON.parse(defaultSelectInstance),
        });
      }
    }
  };

  useEffect(() => {
    // 管理员登录时候不获取实例列表
    if (!state.isAdminLogin) {
      getInstanceList();
    }
  }, [state.isAdminLogin]);

  useEffect(() => {
    setState({
      ...state,
      isAdminLogin: isAdminLogin,
    });
  }, [isAdminLogin]);

  const handleSwitchRole = async () => {
    const resp = await switchUserRole();
    // 切换角色成功后，修改 isAdminLogin 的值，并且重新加载页面
    if (resp.code === "SUCCESS") {
      // 清除之前缓存的实例
      localStorage.removeItem("GEAFLOW_CURRENT_INSTANCE");
      localStorage.removeItem("QUICK_INSTALL_PARAMS");

      const adminStatus = localStorage.getItem("IS_GEAFLOW_ADMIN");
      if (adminStatus) {
        // 已经是管理员，则需要切换成非管理员
        localStorage.removeItem("IS_GEAFLOW_ADMIN");

        // Openpiece 角色切换为 member
        history.push("/studio");

        setState({
          ...state,
          isAdminLogin: null,
        });
      } else {
        localStorage.setItem("IS_GEAFLOW_ADMIN", "true");
        // Openpiece 角色切换为 admin
        history.push("/quickInstall");
        setState({
          ...state,
          isAdminLogin: "true",
        });
      }
    } else {
      message.error(`操作失败: ${resp.message}`);
    }
  };

  useEffect(() => {
    // 首次进入页面，如果没有设置过语言，则默认设置为中文
    const currentLanguage = localStorage.getItem("i18nextLng");
    if (!currentLanguage) {
      const defaultLanguage =
        navigator.language === ("en" || "en-US") ? "en-US" : "zh-CN";
      handleSwitchLanguage(defaultLanguage);
    }
  }, []);

  const handleSwitchLanguage = (value: string) => {
    // 切换语言
    localStorage.setItem("i18nextLng", value);
    i18n.change(value);
    location.reload();
  };

  const items = (
    <Menu>
      {state.isAdminLogin ? (
        <Menu.Item onClick={handleSwitchRole}>
          <UserSwitchOutlined style={{ marginRight: 8 }} />
          {t("i18n.key.tenant.mode")}
        </Menu.Item>
      ) : (
        <Menu.Item onClick={handleSwitchRole}>
          <UserSwitchOutlined style={{ marginRight: 8 }} />
          {t("i18n.key.system.mode")}
        </Menu.Item>
      )}

      <Menu.SubMenu
        title={
          <>
            <GlobalOutlined style={{ marginRight: 8 }} />
            {t("i18n.key.switch.language")}
          </>
        }
      >
        <Menu.Item onClick={() => handleSwitchLanguage("zh-CN")}>
          {t("i18n.key.chinese")}
        </Menu.Item>
        <Menu.Item onClick={() => handleSwitchLanguage("en-US")}>
          {t("i18n.key.English")}
        </Menu.Item>
      </Menu.SubMenu>
      <Menu.Item onClick={handleLogout}>
        <LogoutOutlined style={{ marginRight: 8 }} />
        {t("i18n.key.logout")}
      </Menu.Item>
    </Menu>
  );

  const onChangeInstance = (value) => {
    const { key } = value;
    const [k, v] = key.split(",");
    const current = {
      key: k,
      value: v,
    };
    setState({
      ...state,
      currentInstance: current,
    });

    localStorage.setItem("GEAFLOW_CURRENT_INSTANCE", JSON.stringify(current));
    // 切换实例以后，重新加载页面
    window.location.reload();
  };

  const instanceItems = (
    <Menu onClick={onChangeInstance}>
      {state.instanceList.map((item) => {
        return (
          <Menu.Item key={`${item.id},${item.name}`}>{item.name}</Menu.Item>
        );
      })}
    </Menu>
  );

  return (
    <div className={cx(cls["gm-header"])}>
      <div className={cls.right}>
        <div className="gm-header-toolbar">
          {!state.isAdminLogin && (
            <>
              {state.instanceList.length === 0 ? (
                <Button type="text" onClick={(e) => e.preventDefault()}>
                  <Space>
                    <Tag color="red">{t("i18n.key.instance.first")}</Tag>
                  </Space>
                </Button>
              ) : (
                <Dropdown overlay={instanceItems}>
                  <Button type="text" onClick={(e) => e.preventDefault()}>
                    <Space>
                      {state.currentInstance?.value ||
                        t("i18n.key.select.instance")}
                      <DownOutlined />
                    </Space>
                  </Button>
                </Dropdown>
              )}
            </>
          )}

          <Dropdown overlay={items}>
            <Button type="text" onClick={(e) => e.preventDefault()}>
              <Space>
                {t("i18n.key.welcome")}
                {localStorage.getItem("GEAFLOW_LOGIN_USERNAME")}
                <DownOutlined />
              </Space>
            </Button>
          </Dropdown>
        </div>
      </div>
    </div>
  );
};
