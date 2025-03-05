// "use client";

// import Link from "next/link";
// import { usePathname } from "next/navigation";
// import { cn } from "@/lib/utils";
// import { Button } from "@/components/ui/button";
// import { Activity, Home, LineChart } from "lucide-react";
// import { ModeToggle } from "@/components/mode-toggle";

// export default function Navbar() {
//   const pathname = usePathname();

//   const navItems = [
//     {
//       name: "Home",
//       href: "/",
//       icon: <Home className="h-4 w-4 mr-2" />,
//     },
//     {
//       name: "Predict",
//       href: "/predict",
//       icon: <Activity className="h-4 w-4 mr-2" />,
//     },
//     {
//       name: "Statistics",
//       href: "/stats",
//       icon: <LineChart className="h-4 w-4 mr-2" />,
//     },
//   ];

//   return (
//     <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
//       <div className="container flex h-14 p- items-center">
//         <div className="mr-4 flex">
//           <Link href="/" className="flex items-center space-x-2">
//             <Activity className="h-6 w-6" />
//             <span className="font-bold hidden sm:inline-block">
//               MediPredict
//             </span>
//           </Link>
//         </div>
//         <nav className="flex items-center space-x-2 lg:space-x-4 mx-6">
//           {navItems.map((item) => (
//             <Button
//               key={item.href}
//               variant={pathname === item.href ? "default" : "ghost"}
//               size="sm"
//               className={cn(
//                 "h-8",
//                 pathname === item.href
//                   ? "bg-primary text-primary-foreground"
//                   : "text-muted-foreground"
//               )}
//               asChild
//             >
//               <Link href={item.href} className="flex items-center">
//                 {item.icon}
//                 {item.name}
//               </Link>
//             </Button>
//           ))}
//         </nav>
//         <div className="ml-auto flex items-center space-x-4">
//           <ModeToggle />
//         </div>
//       </div>
//     </header>
//   );
// }

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ModeToggle } from "@/components/mode-toggle";
import { Activity, Home, LineChart } from "lucide-react";
import { MobileNav } from "@/components/mobile-nav";

export default function Navbar() {
  const pathname = usePathname();

  const navItems = [
    {
      name: "Home",
      href: "/",
      icon: <Home className="h-4 w-4 mr-2" />,
    },
    {
      name: "Predict",
      href: "/predict",
      icon: <Activity className="h-4 w-4 mr-2" />,
    },
    {
      name: "Statistics",
      href: "/stats",
      icon: <LineChart className="h-4 w-4 mr-2" />,
    },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center px-4">
        <div className="mr-4 flex">
          <Link href="/" className="flex items-center space-x-2">
            <Activity className="h-6 w-6" />
            <span className="font-bold hidden sm:inline-block">
              MediPredict
            </span>
          </Link>
        </div>

        {/* Mobile Navigation */}
        <MobileNav />

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center space-x-2 lg:space-x-4 mx-6">
          {navItems.map((item) => (
            <Button
              key={item.href}
              variant={pathname === item.href ? "default" : "ghost"}
              size="sm"
              className={cn(
                "h-8",
                pathname === item.href
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground"
              )}
              asChild
            >
              <Link href={item.href} className="flex items-center">
                {item.icon}
                {item.name}
              </Link>
            </Button>
          ))}
        </nav>
        <div className="ml-auto flex items-center space-x-4">
          <ModeToggle />
        </div>
      </div>
    </header>
  );
}
