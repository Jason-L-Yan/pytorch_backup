class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        biggest = nums[0]
        numLen = len(nums)
        for i in range(numLen):
            sum = nums[i]
            if sum > biggest:
                biggest = sum
            for j in range(i + 1, numLen):
                sum += nums[j]
                if sum > biggest:
                    biggest = sum
        return biggest 



print(Solution().maxSubArray([-2, 1]))
