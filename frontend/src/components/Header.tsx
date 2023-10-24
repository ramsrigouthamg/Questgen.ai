import React from "react";
import LogoHome from "../assets/svg/LogoHome";
import "../assets/css/Header.css"
export default function Header() {
  return (
    <div className="hearder flex-space-between flex-direction-column bg-hearder font-size-20px color-white h-full">
      <div className="flex flex-row flex-space-between h-auto items-center font-medium  py-1 text-white">
          <LogoHome/>
          <div>Home</div>
      </div>
      <div className="flex-end">
      </div>
    </div>
  )
}