/**
 * Public Researcher Leaderboard Page
 * Gamification leaderboard with Glicko-2 ratings and badges
 */

import React, { useState, useEffect } from 'react';
import { NextPage } from 'next';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Badge,
  Button,
  Progress,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  Avatar,
  AvatarFallback,
  AvatarImage,
} from '@/components/ui';
import {
  Trophy,
  Medal,
  Award,
  TrendingUp,
  Star,
  Shield,
  Target,
  Zap,
  Crown,
  Gem,
} from 'lucide-react';
import { format } from 'date-fns';

interface LeaderboardEntry {
  rank: number;
  researcher_id?: string;
  handle: string;
  rating: number;
  rd: number;
  tier: string;
  total_findings: number;
  accepted_findings: number;
  total_earnings: number;
  xp_multiplier: number;
  days_since_activity: number;
  rating_confidence: string;
}

interface TierStats {
  tier: string;
  count: number;
  avg_rating: number;
  avg_earnings: number;
}

const LeaderboardPage: NextPage = () => {
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [tierStats, setTierStats] = useState<TierStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('leaderboard');
  const [includeAnonymous, setIncludeAnonymous] = useState(true);
  const [limit, setLimit] = useState(50);

  useEffect(() => {
    fetchLeaderboardData();
  }, [includeAnonymous, limit]);

  const fetchLeaderboardData = async () => {
    try {
      setLoading(true);
      
      const response = await fetch(
        `/api/gamification/leaderboard?limit=${limit}&include_anonymous=${includeAnonymous}`
      );
      
      if (response.ok) {
        const data = await response.json();
        setLeaderboard(data);
        calculateTierStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch leaderboard:', error);
    } finally {
      setLoading(false);
    }
  };

  const calculateTierStats = (data: LeaderboardEntry[]) => {
    const tierGroups = data.reduce((acc, entry) => {
      if (!acc[entry.tier]) {
        acc[entry.tier] = [];
      }
      acc[entry.tier].push(entry);
      return acc;
    }, {} as Record<string, LeaderboardEntry[]>);

    const stats = Object.entries(tierGroups).map(([tier, entries]) => ({
      tier,
      count: entries.length,
      avg_rating: entries.reduce((sum, e) => sum + e.rating, 0) / entries.length,
      avg_earnings: entries.reduce((sum, e) => sum + e.total_earnings, 0) / entries.length,
    }));

    stats.sort((a, b) => b.avg_rating - a.avg_rating);
    setTierStats(stats);
  };

  const getTierIcon = (tier: string) => {
    switch (tier) {
      case 'Master':
        return <Crown className="h-5 w-5 text-red-500" />;
      case 'Diamond':
        return <Gem className="h-5 w-5 text-blue-400" />;
      case 'Platinum':
        return <Star className="h-5 w-5 text-gray-300" />;
      case 'Gold':
        return <Medal className="h-5 w-5 text-yellow-500" />;
      case 'Silver':
        return <Shield className="h-5 w-5 text-gray-400" />;
      case 'Bronze':
        return <Target className="h-5 w-5 text-amber-600" />;
      default:
        return <Award className="h-5 w-5 text-gray-500" />;
    }
  };

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'Master':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'Diamond':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'Platinum':
        return 'bg-gray-100 text-gray-800 border-gray-200';
      case 'Gold':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'Silver':
        return 'bg-slate-100 text-slate-800 border-slate-200';
      case 'Bronze':
        return 'bg-amber-100 text-amber-800 border-amber-200';
      default:
        return 'bg-gray-100 text-gray-600 border-gray-200';
    }
  };

  const getRankIcon = (rank: number) => {
    if (rank === 1) {
      return <Trophy className="h-6 w-6 text-yellow-500" />;
    } else if (rank === 2) {
      return <Medal className="h-6 w-6 text-gray-400" />;
    } else if (rank === 3) {
      return <Award className="h-6 w-6 text-amber-600" />;
    }
    return null;
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const getActivityStatus = (daysSince: number) => {
    if (daysSince === 0) return { text: 'Active today', color: 'text-green-600' };
    if (daysSince <= 7) return { text: `${daysSince}d ago`, color: 'text-green-500' };
    if (daysSince <= 30) return { text: `${daysSince}d ago`, color: 'text-yellow-500' };
    return { text: `${daysSince}d ago`, color: 'text-red-500' };
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        <span className="ml-2">Loading leaderboard...</span>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight flex items-center justify-center gap-2">
            <Trophy className="h-8 w-8 text-yellow-500" />
            Researcher Leaderboard
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Top security researchers ranked by Glicko-2 rating system. 
            Earn points by discovering vulnerabilities and climb the ranks!
          </p>
        </div>

        {/* Controls */}
        <div className="flex justify-center space-x-4">
          <Button
            variant={includeAnonymous ? "default" : "outline"}
            onClick={() => setIncludeAnonymous(!includeAnonymous)}
          >
            {includeAnonymous ? "Hide" : "Show"} Anonymous
          </Button>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
            <TabsTrigger value="tiers">Tier Statistics</TabsTrigger>
          </TabsList>

          <TabsContent value="leaderboard" className="space-y-4">
            {/* Top 3 Podium */}
            {leaderboard.length >= 3 && (
              <Card className="bg-gradient-to-r from-yellow-50 to-amber-50 border-yellow-200">
                <CardHeader>
                  <CardTitle className="text-center">Top Researchers</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    {/* 2nd Place */}
                    <div className="text-center space-y-2 order-1">
                      <div className="relative">
                        <Avatar className="w-16 h-16 mx-auto border-4 border-gray-300">
                          <AvatarFallback className="bg-gray-100 text-gray-600 text-lg font-bold">
                            {leaderboard[1].handle.substring(0, 2).toUpperCase()}
                          </AvatarFallback>
                        </Avatar>
                        <div className="absolute -top-2 -right-2">
                          <Medal className="h-8 w-8 text-gray-400" />
                        </div>
                      </div>
                      <div>
                        <p className="font-semibold">{leaderboard[1].handle}</p>
                        <Badge className={getTierColor(leaderboard[1].tier)}>
                          {getTierIcon(leaderboard[1].tier)}
                          <span className="ml-1">{leaderboard[1].tier}</span>
                        </Badge>
                        <p className="text-sm text-muted-foreground">
                          {Math.round(leaderboard[1].rating)} rating
                        </p>
                      </div>
                    </div>

                    {/* 1st Place */}
                    <div className="text-center space-y-2 order-2">
                      <div className="relative">
                        <Avatar className="w-20 h-20 mx-auto border-4 border-yellow-400">
                          <AvatarFallback className="bg-yellow-100 text-yellow-800 text-xl font-bold">
                            {leaderboard[0].handle.substring(0, 2).toUpperCase()}
                          </AvatarFallback>
                        </Avatar>
                        <div className="absolute -top-2 -right-2">
                          <Trophy className="h-10 w-10 text-yellow-500" />
                        </div>
                      </div>
                      <div>
                        <p className="font-bold text-lg">{leaderboard[0].handle}</p>
                        <Badge className={getTierColor(leaderboard[0].tier)}>
                          {getTierIcon(leaderboard[0].tier)}
                          <span className="ml-1">{leaderboard[0].tier}</span>
                        </Badge>
                        <p className="text-sm text-muted-foreground">
                          {Math.round(leaderboard[0].rating)} rating
                        </p>
                        {leaderboard[0].xp_multiplier > 1 && (
                          <Badge variant="secondary" className="text-xs">
                            <Zap className="h-3 w-3 mr-1" />
                            {leaderboard[0].xp_multiplier}x XP
                          </Badge>
                        )}
                      </div>
                    </div>

                    {/* 3rd Place */}
                    <div className="text-center space-y-2 order-3">
                      <div className="relative">
                        <Avatar className="w-16 h-16 mx-auto border-4 border-amber-600">
                          <AvatarFallback className="bg-amber-100 text-amber-800 text-lg font-bold">
                            {leaderboard[2].handle.substring(0, 2).toUpperCase()}
                          </AvatarFallback>
                        </Avatar>
                        <div className="absolute -top-2 -right-2">
                          <Award className="h-8 w-8 text-amber-600" />
                        </div>
                      </div>
                      <div>
                        <p className="font-semibold">{leaderboard[2].handle}</p>
                        <Badge className={getTierColor(leaderboard[2].tier)}>
                          {getTierIcon(leaderboard[2].tier)}
                          <span className="ml-1">{leaderboard[2].tier}</span>
                        </Badge>
                        <p className="text-sm text-muted-foreground">
                          {Math.round(leaderboard[2].rating)} rating
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Full Leaderboard Table */}
            <Card>
              <CardHeader>
                <CardTitle>Full Rankings</CardTitle>
                <CardDescription>
                  Complete leaderboard with detailed statistics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-16">Rank</TableHead>
                      <TableHead>Researcher</TableHead>
                      <TableHead>Rating</TableHead>
                      <TableHead>Tier</TableHead>
                      <TableHead>Findings</TableHead>
                      <TableHead>Earnings</TableHead>
                      <TableHead>Activity</TableHead>
                      <TableHead>Bonus</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {leaderboard.map((entry) => {
                      const activity = getActivityStatus(entry.days_since_activity);
                      
                      return (
                        <TableRow key={entry.rank} className={entry.rank <= 3 ? 'bg-yellow-50/50' : ''}>
                          <TableCell className="font-medium">
                            <div className="flex items-center space-x-2">
                              {getRankIcon(entry.rank)}
                              <span className={entry.rank <= 3 ? 'font-bold' : ''}>
                                #{entry.rank}
                              </span>
                            </div>
                          </TableCell>
                          
                          <TableCell>
                            <div className="flex items-center space-x-3">
                              <Avatar className="w-8 h-8">
                                <AvatarFallback className="text-sm">
                                  {entry.handle.substring(0, 2).toUpperCase()}
                                </AvatarFallback>
                              </Avatar>
                              <div>
                                <p className="font-medium">{entry.handle}</p>
                                <p className="text-xs text-muted-foreground">
                                  RD: {Math.round(entry.rd)} • {entry.rating_confidence}
                                </p>
                              </div>
                            </div>
                          </TableCell>
                          
                          <TableCell>
                            <div className="font-medium">
                              {Math.round(entry.rating)}
                            </div>
                          </TableCell>
                          
                          <TableCell>
                            <Badge className={getTierColor(entry.tier)}>
                              {getTierIcon(entry.tier)}
                              <span className="ml-1">{entry.tier}</span>
                            </Badge>
                          </TableCell>
                          
                          <TableCell>
                            <div className="text-sm">
                              <p className="font-medium">{entry.accepted_findings}</p>
                              <p className="text-muted-foreground">
                                of {entry.total_findings}
                              </p>
                            </div>
                          </TableCell>
                          
                          <TableCell>
                            <div className="font-medium">
                              {formatCurrency(entry.total_earnings)}
                            </div>
                          </TableCell>
                          
                          <TableCell>
                            <span className={`text-sm ${activity.color}`}>
                              {activity.text}
                            </span>
                          </TableCell>
                          
                          <TableCell>
                            {entry.xp_multiplier > 1 && (
                              <Badge variant="secondary" className="text-xs">
                                <Zap className="h-3 w-3 mr-1" />
                                {entry.xp_multiplier}x
                              </Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="tiers" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {tierStats.map((tier) => (
                <Card key={tier.tier}>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">
                      {tier.tier} Tier
                    </CardTitle>
                    {getTierIcon(tier.tier)}
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{tier.count}</div>
                    <p className="text-xs text-muted-foreground">researchers</p>
                    
                    <div className="mt-4 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Avg Rating:</span>
                        <span className="font-medium">
                          {Math.round(tier.avg_rating)}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Avg Earnings:</span>
                        <span className="font-medium">
                          {formatCurrency(tier.avg_earnings)}
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Tier Distribution Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Tier Distribution</CardTitle>
                <CardDescription>
                  Percentage of researchers in each tier
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {tierStats.map((tier) => {
                    const percentage = (tier.count / leaderboard.length) * 100;
                    
                    return (
                      <div key={tier.tier} className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <div className="flex items-center space-x-2">
                            {getTierIcon(tier.tier)}
                            <span className="font-medium">{tier.tier}</span>
                          </div>
                          <span>{tier.count} ({percentage.toFixed(1)}%)</span>
                        </div>
                        <Progress value={percentage} className="h-2" />
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default LeaderboardPage;