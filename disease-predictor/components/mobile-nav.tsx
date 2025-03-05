"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Activity, Home, LineChart, Menu } from "lucide-react";

export function MobileNav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const navItems = [
    {
      name: "Home",
      href: "/",
      icon: <Home className="h-5 w-5 mr-2" />,
    },
    {
      name: "Predict",
      href: "/predict",
      icon: <Activity className="h-5 w-5 mr-2" />,
    },
    {
      name: "Statistics",
      href: "/stats",
      icon: <LineChart className="h-5 w-5 mr-2" />,
    },
  ];

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="md:hidden">
          <Menu className="h-5 w-5" />
          <span className="sr-only">Toggle menu</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="w-[240px] sm:w-[300px]">
        <div className="flex flex-col space-y-6 py-4">
          <div className="flex items-center space-x-2 px-2">
            <Activity className="h-6 w-6 text-primary" />
            <span className="font-bold text-lg">MediPredict</span>
          </div>
          <nav className="flex flex-col space-y-2">
            {navItems.map((item) => (
              <Button
                key={item.href}
                variant={pathname === item.href ? "default" : "ghost"}
                className={cn(
                  "justify-start",
                  pathname === item.href
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground"
                )}
                asChild
                onClick={() => setOpen(false)}
              >
                <Link href={item.href} className="flex items-center">
                  {item.icon}
                  {item.name}
                </Link>
              </Button>
            ))}
          </nav>
        </div>
      </SheetContent>
    </Sheet>
  );
}
