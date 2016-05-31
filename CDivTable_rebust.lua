local CDivTable_robust, parent = torch.class('CDivTable_robust', 'nn.Module')

function CDivTable_robust:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CDivTable_robust:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])   
   self.output:cdiv(input[2])
   self.output[input[2]:abs():le(0.1)] = 0.0
   --print('output '..self.output:norm().. ' '..input[1]:norm()..' '..input[2]:norm())
   return self.output
end

function CDivTable_robust:updateGradInput(input, gradOutput)
   -- print('gradoutput '..gradOutput:norm())
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[2] = self.gradInput[2] or input[1].new()

   tmp = input[2]:abs():le(0.1)

   self.gradInput[1]:resizeAs(input[1]):copy(gradOutput):cdiv(input[2])
   self.gradInput[2]:resizeAs(input[2]):zero():addcdiv(-1,self.gradInput[1],input[2]):cmul(input[1])
   
   self.gradInput[1][tmp:eq(1)] = 0.0 --do not backpropagate nan when you see zero/zero
   self.gradInput[2][tmp:eq(1)] = 0.0
   -- self.gradInput[1][self.gradInput[1]:gt(10)]=10
   -- self.gradInput[1][self.gradInput[1]:lt(-10)]=-10
   -- self.gradInput[2][self.gradInput[2]:gt(10)]=10
   -- self.gradInput[2][self.gradInput[2]:lt(-10)]=-10   

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   --print('gradinput: '..self.gradInput[1]:norm() .. ' '..self.gradInput[2]:norm())
   return self.gradInput
end