import { Tabs } from "expo-router";
import { Ionicons } from "@expo/vector-icons";

import { fonts, fontSizes, useTheme } from "../../lib/theme";

export default function TabsLayout() {
  const { colors } = useTheme(); // system-adaptive theme, task 2 (2026-08-20)
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: colors.navy,
        tabBarInactiveTintColor: colors.textFaint,
        tabBarStyle: { backgroundColor: colors.surface, borderTopColor: colors.border },
        headerStyle: { backgroundColor: colors.surface },
        headerShadowVisible: false,
        headerTitleStyle: { fontFamily: fonts.serif, fontSize: fontSizes.h2, color: colors.navyInk },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Home",
          tabBarIcon: ({ color, size, focused }) => (
            <Ionicons name={focused ? "home" : "home-outline"} size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="history"
        options={{
          title: "History",
          // The History route has its own nested Stack (history/_layout.js) which
          // renders its own header for both the list and detail screens -- without
          // this, the tab bar's header and the nested Stack's header both render
          // "History" stacked on top of each other.
          headerShown: false,
          tabBarIcon: ({ color, size, focused }) => (
            <Ionicons name={focused ? "list" : "list-outline"} size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: "Settings",
          tabBarIcon: ({ color, size, focused }) => (
            <Ionicons name={focused ? "settings-sharp" : "settings-outline"} size={size} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}
