import QuantLib as ql
import numpy as np

# path generator for a given 1d stochastic process and 
# a given set of QuantLib dates, which can be unevenly distributed
# uses process evolve method, which returns asset value after time interval Δt
# returns E(x0,t0,Δt) + S(x0,t0,Δt) ⋅ Δw, where E is expectation and S standard deviation
# input arguments:
#   x0 = asset value at inception
#   dates = array of dates
#   dayCounter = QuantLib day counter
#   process = QuantLib 1D stochastic process implementation
#   nPaths = number of paths to be simulated
def PathGenerator(x0, dates, dayCounter, process, nPaths):
    t = np.array([dayCounter.yearFraction(dates[0], d) for d in dates])    
    urg = ql.UniformRandomGenerator()
    ursg = ql.UniformRandomSequenceGenerator(t.shape[0] - 1, urg)
    grsg = ql.GaussianRandomSequenceGenerator(ursg)    
    paths = np.zeros(shape = (nPaths, t.shape[0]))
    
    for j in range(nPaths):
        dw = np.array(list(grsg.nextSequence().value()))
        x = x0
        path = []

        for i in range(1, t.shape[0]):
            x = process.evolve(t[i-1], x, (t[i] - t[i-1]), dw[i-1])
            path.append(x)
            
        path = np.hstack([np.array([x0]), np.array(path)])
        paths[j,:] = path
        
    # return array dimensions: [number of paths, number of items in t array]
    return paths


def TARN(bondStartDate, valuationDate, couponDates, targetCoupon, teaserCoupon, cap, floor,
    hasToReachTarget, payoff, notional, dayCounter, nPaths, process, curve, index):    
    
    # immediate exit trigger for matured transaction
    if(valuationDate >= couponDates[-1]):
        return 0.0, 0.0
    
    # create date array for path generator
    # combine valuation date and all remaining coupon dates
    dates = np.hstack((np.array([valuationDate]), couponDates[couponDates > valuationDate]))
    
    # generate paths for a given set of dates, exclude the current spot rate
    paths = PathGenerator(process.x0(), dates, dayCounter, process, nPaths)[:,1:]
    
    # identify past coupon dates
    pastDates = couponDates[couponDates <= valuationDate]
    # conditionally, merge given past fixings and generated paths
    if(pastDates.shape[0] > 0):
        pastFixings = np.array([index.fixing(pastDate) for pastDate in pastDates])    
        pastFixings = np.tile(pastFixings, (paths.shape[0], 1))
        paths = np.hstack((pastFixings, paths))
        
    # define time grid for all coupon dates, calculate day count fractions
    t = np.array([0.0] + [dayCounter.yearFraction(bondStartDate, d) for d in couponDates])
    dcf = np.diff(t)
    
    # result accumulators
    global_pv = []
    termination = []

    # calculate PV for all paths
    for path in paths:
        # transform simulated path into structured rates using payoff function
        path = (np.vectorize(payoff))(path)
        index_1 = np.where(teaserCoupon > 0.0)
        # replace some path rates with teaser coupons (if exists)
        path[index_1] = teaserCoupon
        # calculate capped and floored structured coupon for non-teaser rates
        path = np.concatenate([path[index_1], np.minimum(path[index_1[0].shape[0]:], cap)])
        path = np.concatenate([path[index_1], np.maximum(path[index_1[0].shape[0]:], floor)])
        # multiply rates with day count fractions
        path = np.multiply(path, dcf)
        # take into account only rates, for which cumulative sum is less or equal to target coupon
        index_2 = np.where(np.cumsum(path) <= targetCoupon)
        path = path[index_2]
        dates = couponDates[index_2]
        # path termination time is the date, which reaches target coupon
        termination.append(dayCounter.yearFraction(valuationDate, dates[-1]))
        # if coupon has to reach target coupon, add remaining coupon available into final coupon
        if(hasToReachTarget): path[-1] = targetCoupon - np.sum(path[:-1])
        # multiply coupon rates with notionals, add final redemption
        path *= notional
        path[-1] += notional
        # take into account only coupons, for which coupon dates are in the future
        index_3 = np.where(dates >= valuationDate)
        dates = dates[index_3]
        path = path[index_3]
        # request discount factors for all coupon dates
        df = np.array([curve.discount(d) for d in dates])
        # calculate coupon PV's
        path = np.multiply(path, df)
        # add path PV into result accumulator
        global_pv.append(np.sum(path))

    # return tuple (pv, average termination time)
    return (np.mean(np.array(global_pv)), np.mean(np.array(termination)))

# define general QuantLib-related parameters
valuationDate = ql.Date(4,3,2018)
calendar = ql.TARGET()
convention = ql.ModifiedFollowing
dayCounter = ql.Actual360()
ql.Settings.instance().evaluationDate = valuationDate

# set index object and past fixings
pastFixingsDates = np.array([ql.Date(4,3,2019), ql.Date(4,3,2020)])
pastFixingsRates = np.array([0.05, 0.05])
index = ql.USDLibor(ql.Period(12, ql.Months))
index.clearFixings()
index.addFixings(pastFixingsDates, pastFixingsRates)

# create discounting curve and process for short rate
r0 = 0.015
curveHandle = ql.YieldTermStructureHandle(ql.FlatForward(valuationDate, r0, dayCounter))
a = 0.05
vol = 0.009
HW1F = ql.HullWhiteProcess(curveHandle, a, vol)

# define bond-related parameters
startDate = ql.Date(4,3,2018)
firstCouponDate = calendar.advance(startDate, ql.Period(1, ql.Years))
lastCouponDate = calendar.advance(startDate, ql.Period(10, ql.Years))
couponDates = np.array(list(ql.Schedule(firstCouponDate, lastCouponDate, ql.Period(ql.Annual), 
    calendar, ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Forward, False)))
teaserCoupon = np.array([0.1])
targetCoupon = 0.25
hasToReachTarget = True
cap = 0.15
floor = 0.0
fixedRate = 0.1
factor = 3.0
structuredCouponPayoff = lambda r: max(fixedRate - factor * r, 0.0)
notional = 1000000.0

# define monte carlo-related parameters
nPaths = 10000

# request pv
result = TARN(startDate, valuationDate, couponDates, targetCoupon, teaserCoupon,
    cap, floor, hasToReachTarget, structuredCouponPayoff, notional, dayCounter,
    nPaths, HW1F, curveHandle, index)

print('pv', '{0:.0f}'.format(result[0]))
print('termination', '{0:.1f}'.format(result[1]))
